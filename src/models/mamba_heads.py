"""
Task heads for finetuning the pretrained Bi-Mamba encoder on the MSA
benchmark suite (see docs/BENCHMARKING.md). Mirrors the split already used
by MambaForMaskedLM in mamba.py: a shared MambaEncoder trunk + a thin,
task-specific head on top, so the same pretrained checkpoint can be reused
across sequence classification, token classification, and extractive QA
without touching the encoder implementation itself.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba import MambaEncoder, MambaForMaskedLM


def load_pretrained_encoder(encoder, ckpt_path, device="cpu", strict=False):
    """Loads encoder.* weights from a MambaForMaskedLM training checkpoint
    (as saved by train.py: ckpt["model"] is model.module.state_dict()).
    Drops mlm_head.*/decoder.* keys since those don't exist on task heads.
    strict=False by default: the tied decoder + mlm_head are *expected* to
    be absent here, that's not a real mismatch, but we still print
    whatever comes back so a genuinely wrong checkpoint isn't silently
    swallowed.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    encoder_state = {
        k[len("encoder.") :]: v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError(
            f"no 'encoder.*' keys found in {ckpt_path} -- is this really a "
            "MambaForMaskedLM training checkpoint?"
        )
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=strict)
    if missing:
        print(f"[load_pretrained_encoder] missing keys: {missing}")
    if unexpected:
        print(f"[load_pretrained_encoder] unexpected keys: {unexpected}")
    return encoder


def _mean_pool(hidden, attention_mask):
    """Mean-pool over real (non-padding) positions. Preferred over a
    CLS-token pool here: unlike BERT, this encoder was never pretrained
    with a dedicated [CLS] aggregation objective (MLM only), so there's no
    reason to expect position 0 to hold a good sentence summary. Mean
    pooling over the bidirectional Mamba output is the safer default for
    a from-scratch architecture."""
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


class MambaForSequenceClassification(nn.Module):
    """Sentence(-pair) classification: sentiment (HARD), NLI (XNLI-ar),
    topic/news classification (SANAD/ASND). For sentence-pair tasks,
    concatenate premise + [SEP] + hypothesis before tokenizing -- the
    tokenizer owns sequence packing here, same convention as the rest of
    this codebase, the model itself stays pair-agnostic."""

    def __init__(self, config, num_labels, dropout=None):
        super().__init__()
        self.num_labels = num_labels
        self.encoder = MambaEncoder(config)
        d_model = config["d_model"]
        self.dropout = nn.Dropout(
            dropout if dropout is not None else config.get("dropout", 0.1)
        )
        self.classifier = nn.Linear(d_model, num_labels)
        self.classifier.apply(MambaForMaskedLM._init_weights)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.encoder(input_ids, attention_mask)
        pooled = (
            _mean_pool(hidden, attention_mask)
            if attention_mask is not None
            else hidden.mean(dim=1)
        )
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


class MambaForTokenClassification(nn.Module):
    """Token-level classification: NER (ANERcorp), POS tagging. labels use
    -100 for positions to ignore (padding, and for wordpiece tokenizers,
    non-first subword pieces of a word) -- same ignore_index convention as
    the MLM head in mamba.py, kept consistent on purpose."""

    def __init__(self, config, num_labels, dropout=None):
        super().__init__()
        self.num_labels = num_labels
        self.encoder = MambaEncoder(config)
        d_model = config["d_model"]
        self.dropout = nn.Dropout(
            dropout if dropout is not None else config.get("dropout", 0.1)
        )
        self.classifier = nn.Linear(d_model, num_labels)
        self.classifier.apply(MambaForMaskedLM._init_weights)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.encoder(input_ids, attention_mask)
        logits = self.classifier(self.dropout(hidden))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels), labels.view(-1), ignore_index=-100
            )
        return {"loss": loss, "logits": logits}


class MambaForQuestionAnswering(nn.Module):
    """Extractive QA (ARCD): predicts start/end token indices of the
    answer span within a packed [question] [SEP] [context] sequence. Two
    independent linear heads on the shared encoder output -- same
    span-prediction formulation used for BERT-on-SQuAD."""

    def __init__(self, config):
        super().__init__()
        self.encoder = MambaEncoder(config)
        d_model = config["d_model"]
        self.qa_outputs = nn.Linear(d_model, 2)
        self.qa_outputs.apply(MambaForMaskedLM._init_weights)

    def forward(
        self, input_ids, attention_mask=None, start_positions=None, end_positions=None
    ):
        hidden = self.encoder(input_ids, attention_mask)
        logits = self.qa_outputs(hidden)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            start_loss = F.cross_entropy(
                start_logits, start_positions, ignore_index=ignored_index
            )
            end_loss = F.cross_entropy(
                end_logits, end_positions, ignore_index=ignored_index
            )
            loss = (start_loss + end_loss) / 2

        return {"loss": loss, "start_logits": start_logits, "end_logits": end_logits}
