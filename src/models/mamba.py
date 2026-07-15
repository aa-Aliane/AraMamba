"""
Bidirectional Mamba encoder for Arabic MLM pretraining (AraBERT-style task,
Mamba/SSM backbone instead of Transformer attention).

Mamba itself is causal. To get a BERT-like encoder we run a forward mixer
and a mixer on the reversed sequence in every block, then merge -> full
bidirectional context, still O(L) instead of O(L^2).

Uses `mamba_ssm` CUDA kernels if importable (fast, needs mamba-ssm +
causal-conv1d installed and building against your CUDA/torch version).
Falls back to a slow pure-PyTorch scan otherwise so the code still runs
if the CUDA kernels fail to build on your 2080Tis -- use the fallback only
to sanity-check correctness, not for real training (it's ~10-50x slower).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba

    MAMBA_SSM_AVAILABLE = True
except ImportError:
    MAMBA_SSM_AVAILABLE = False


class NaiveSSM(nn.Module):
    """Pure-PyTorch selective-scan fallback. Slow (python loop over L)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1)
        self.dt_proj = nn.Linear(1, self.d_inner)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float()).repeat(self.d_inner, 1)
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):
        B, L, _ = x.shape
        x_in, z = self.in_proj(x).chunk(2, dim=-1)
        x_conv = self.conv1d(x_in.transpose(1, 2))[..., :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        dt, B_ssm, C_ssm = torch.split(
            self.x_proj(x_conv), [1, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t, :].unsqueeze(-1) * A.unsqueeze(0))
            dB = dt[:, t, :].unsqueeze(-1) * B_ssm[:, t, :].unsqueeze(1)
            h = h * dA + dB * x_conv[:, t, :].unsqueeze(-1)
            ys.append((h * C_ssm[:, t, :].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + x_conv * self.D
        y = y * F.silu(z)
        return self.out_proj(y)


def _make_mixer(d_model, d_state, d_conv, expand):
    if MAMBA_SSM_AVAILABLE:
        return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    return NaiveSSM(d_model, d_state, d_conv, expand)


class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd_mixer = _make_mixer(d_model, d_state, d_conv, expand)
        self.bwd_mixer = _make_mixer(d_model, d_state, d_conv, expand)
        self.merge = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x, attention_mask=None):
        residual = x
        h = self.norm(x)
        fwd = self.fwd_mixer(h)
        bwd_in = h.flip(dims=[1])
        if attention_mask is not None:
            bwd_in = bwd_in * attention_mask.flip(dims=[1]).unsqueeze(-1)
        bwd = self.bwd_mixer(bwd_in).flip(dims=[1])
        merged = self.merge(torch.cat([fwd, bwd], dim=-1))
        x = residual + self.dropout(merged)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class MambaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config["d_model"]
        self.word_emb = nn.Embedding(
            config["vocab_size"], d_model, padding_idx=config.get("pad_token_id", 0)
        )
        self.pos_emb = nn.Embedding(config["max_position_embeddings"], d_model)
        self.emb_dropout = nn.Dropout(config.get("dropout", 0.1))
        self.layers = nn.ModuleList(
            [
                BiMambaBlock(
                    d_model,
                    config["d_state"],
                    config["d_conv"],
                    config["expand"],
                    config.get("dropout", 0.1),
                )
                for _ in range(config["n_layer"])
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.gradient_checkpointing = config.get("gradient_checkpointing", False)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.word_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, attention_mask, use_reentrant=False
                )
            else:
                x = layer(x, attention_mask)
        return self.final_norm(x)


class MambaForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = MambaEncoder(config)
        d_model = config["d_model"]
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.decoder = nn.Linear(d_model, config["vocab_size"], bias=True)
        self.decoder.weight = self.encoder.word_emb.weight  # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        """BERT-style init: small-std normal for Linear/Embedding weights,
        zeroed biases, zeroed padding row. Left at PyTorch defaults (std=1
        for nn.Embedding) this model starts with wildly overconfident,
        miscalibrated logits over the 64k vocab, producing a much higher
        initial MLM loss than the ~ln(vocab_size) expected from random
        guessing, and unstable early training."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.encoder(input_ids, attention_mask)
        logits = self.decoder(self.mlm_head(hidden))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
        return {"loss": loss, "logits": logits, "hidden_states": hidden}
