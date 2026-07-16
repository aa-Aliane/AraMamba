# Tree View:
```
.
├── configs
│   ├── base.yaml
│   ├── dry_test.yaml
│   ├── eval_check.yaml
│   └── pilot.yaml
└── src
    ├── data
    │   └── prepare_data.py
    ├── models
    │   ├── __pycache__
    │   │   └── mamba.cpython-311.pyc
    │   └── mamba.py
    ├── train.py
    └── utils
        ├── __init__.py
        ├── __pycache__
        │   ├── __init__.cpython-311.pyc
        │   └── eval.cpython-311.pyc
        └── eval.py

```

# Content:

## configs/base.yaml

```yaml
model:
  d_model: 512
  n_layer: 12
  d_state: 16
  d_conv: 4
  expand: 2
  max_position_embeddings: 512
  dropout: 0.1
  pad_token_id: 0
  gradient_checkpointing: true   # required to fit on 11GB cards

training:
  batch_size: 8
  grad_accum_steps: 8
  lr: 3.0e-4
  weight_decay: 0.01
  warmup_steps: 10000
  max_steps: 200000
  mlm_probability: 0.15
  fp16: true                    # Turing (2080Ti) has no real bf16 support, use fp16
  seed: 42
  save_steps: 5000
  log_steps: 100
  num_workers: 8

data:
  tokenizer_name: "aubmindlab/bert-base-arabertv02"  # vocab_size=64000, taken from tokenizer at runtime
  train_dir: "outputs/data/train"   # dir of shard_*.txt, produced by prepare_data.py
  val_dir: "outputs/data/val"
  max_seq_length: 512

paths:
  output_dir: "outputs"
  checkpoint_dir: "checkpoint"
  logs_dir: "logs"
  results_dir: "results"

distributed:
  num_gpus: 4
  backend: "nccl"

```


## configs/dry_test.yaml

```yaml
model:
  d_model: 128
  n_layer: 2
  d_state: 16
  d_conv: 4
  expand: 2
  max_position_embeddings: 512
  dropout: 0.1
  pad_token_id: 0
  gradient_checkpointing: false

training:
  batch_size: 2
  grad_accum_steps: 1
  lr: 3.0e-4
  weight_decay: 0.01
  warmup_steps: 5
  max_steps: 20
  mlm_probability: 0.15
  fp16: true
  seed: 42
  save_steps: 10
  log_steps: 1
  num_workers: 2

distributed:
  num_gpus: 1

data:
  tokenizer_name: "aubmindlab/bert-base-arabertv02"
  train_dir: "outputs/data/train"
  val_dir: "outputs/data/val"
  max_seq_length: 512

paths:
  output_dir: "outputs"
  checkpoint_dir: "checkpoint"
  logs_dir: "logs"
  results_dir: "results"

```


## configs/eval_check.yaml

```yaml
model:
  d_model: 512
  n_layer: 12
  d_state: 16
  d_conv: 4
  expand: 2
  max_position_embeddings: 512
  dropout: 0.1
  pad_token_id: 0
  gradient_checkpointing: true   # same as base: required to fit on 11GB cards

training:
  batch_size: 8                 # same as base: real VRAM/throughput behavior
  grad_accum_steps: 8
  lr: 3.0e-4
  weight_decay: 0.01
  warmup_steps: 10000
  max_steps: 60                 # just enough to cross save_steps a couple times
  mlm_probability: 0.15
  fp16: true
  seed: 42
  save_steps: 10                # hits the eval/checkpoint branch at step 10, 20, 30
  log_steps: 1
  num_workers: 8

data:
  tokenizer_name: "aubmindlab/bert-base-arabertv02"
  train_dir: "outputs/data/train"   # point at a real (even if small) shard dir
  val_dir: "outputs/data/val"
  max_seq_length: 512

paths:
  output_dir: "outputs_eval_check"
  checkpoint_dir: "checkpoint_eval_check"
  logs_dir: "logs_eval_check"
  results_dir: "results_eval_check"

distributed:
  num_gpus: 4
  backend: "nccl"

```


## configs/pilot.yaml

```yaml
model:
  d_model: 512
  n_layer: 12
  d_state: 16
  d_conv: 4
  expand: 2
  max_position_embeddings: 512
  dropout: 0.1
  pad_token_id: 0
  gradient_checkpointing: true   # Keep this on to test real VRAM limits

training:
  batch_size: 8              # This is what we will stress-test
  grad_accum_steps: 8
  lr: 5.0e-4                    # Slightly aggressive for pilot tests
  weight_decay: 0.01
  warmup_steps: 200             # Short warmup for quick diagnostic
  max_steps: 2000               # We only need 1,000 - 2,000 steps to check trajectory
  mlm_probability: 0.15
  fp16: true
  seed: 42
  save_steps: 500
  log_steps: 10
  num_workers: 4

data:
  tokenizer_name: "aubmindlab/bert-base-arabertv02"
  train_dir: "outputs/data/train"  # Pointing to your 500MB dry-run directory
  val_dir: "outputs/data/val"
  max_seq_length: 512

paths:
  output_dir: "outputs_pilot"
  checkpoint_dir: "checkpoint_pilot"
  logs_dir: "logs_pilot"
  results_dir: "results_pilot"

distributed:
  num_gpus: 4
  backend: "nccl"

```


## src/data/prepare_data.py

```py
"""
Build a large, deduplicated Arabic pretraining corpus sized to fit the
available disk, using two streamed sources (no full raw download needed):

  - Arabic Wikipedia            -> clean, encyclopedic register
  - CulturaX (ar)                -> already cleaned + fuzzy-deduplicated
                                     mC4+OSCAR web corpus, ~158B tokens
                                     available for Arabic, so we only need
                                     to consume a slice of it.

CulturaX one-time setup (required before running this script):
  1. accept the terms at https://huggingface.co/datasets/uonlp/CulturaX
  2. `huggingface-cli login` (paste an HF access token with read access)

This script:
  - cleans docs in parallel across CPU cores (regex cleaning, language
    ratio filter, min-length filter)
  - CHUNKS each cleaned document into ~max_chunk_words-sized pieces before
    writing. Without this, one line == one whole document, and Wikipedia
    articles / CulturaX web pages routinely run to 500-2000+ words. Since
    the training-side tokenizer truncates each line to max_seq_length
    tokens, un-chunked long documents (a) blow past the model's sequence
    length so almost every batch is padded/truncated to the max -> much
    higher, less predictable GPU memory use, and (b) silently throw away
    everything past the first ~512 tokens of any long document, wasting
    most of the cleaned/deduped text you paid to collect.
  - exact-dedups via a bounded-memory Bloom filter (no extra dependency),
    applied per-chunk so repeated boilerplate chunks across documents are
    still caught
  - writes to sharded files (<out_dir>/train/shard_00000.txt, ...) instead
    of one giant file, so downstream code never has to load the full
    corpus into RAM
  - stops once --target_size_gb of cleaned text has been written

Usage:
  python src/data/prepare_data.py --target_size_gb 120 --workers 64
"""

import argparse
import functools
import hashlib
import math
import os
import re
from multiprocessing import Pool

from datasets import load_dataset
from tqdm import tqdm

ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")
ARABIC_CHAR = re.compile(r"[\u0600-\u06FF]")
URL_RE = re.compile(r"http\S+|www\.\S+")
WS_RE = re.compile(r"\s+")

MIN_WORDS = 20
MIN_ARABIC_RATIO = 0.7
SHARD_BYTES = 1 * 1024**3  # ~1GB per shard file

# Word-based chunk size, not token-based: this script has no tokenizer
# dependency, so we approximate. Arabic subword tokenizers (e.g. the
# AraBERT vocab used downstream) commonly run close to ~1 token/word to
# ~1.3 tokens/word depending on register, so 400 words keeps most chunks
# comfortably under a 512-token max_seq_length after the tokenizer adds
# [CLS]/[SEP] and does its own subword splitting. Adjust if your
# tokenizer's token/word ratio differs meaningfully.
MAX_CHUNK_WORDS = 400


def clean_doc(text):
    if not text:
        return None
    text = ARABIC_DIACRITICS.sub("", text)
    text = URL_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    if not text:
        return None
    if len(text.split()) < MIN_WORDS:
        return None
    if len(ARABIC_CHAR.findall(text)) / max(1, len(text)) < MIN_ARABIC_RATIO:
        return None
    return text


def chunk_doc(text, max_words=MAX_CHUNK_WORDS):
    """Split a cleaned document into ~max_words-sized chunks so no single
    training line silently exceeds the model's max_seq_length. Splits on
    whitespace-delimited words (cheap, no tokenizer dependency here); the
    last partial chunk is kept only if it still meets MIN_WORDS."""
    words = text.split()
    if len(words) <= max_words:
        yield text
        return
    for i in range(0, len(words), max_words):
        chunk_words = words[i : i + max_words]
        if len(chunk_words) < MIN_WORDS:
            continue
        yield " ".join(chunk_words)


def clean_and_chunk(text, max_chunk_words=MAX_CHUNK_WORDS):
    """Combines clean_doc + chunk_doc into one function so it can be
    passed to Pool.imap_unordered as a single per-item worker call."""
    cleaned = clean_doc(text)
    if cleaned is None:
        return []
    return list(chunk_doc(cleaned, max_words=max_chunk_words))


class BloomFilter:
    """Dependency-free, bounded-memory exact-dedup filter."""

    def __init__(self, capacity=200_000_000, error_rate=0.001):
        self.size = int(-capacity * math.log(error_rate) / (math.log(2) ** 2))
        self.k = max(1, int((self.size / capacity) * math.log(2)))
        self.bits = bytearray(self.size // 8 + 1)

    def _positions(self, item):
        h = hashlib.blake2b(item.encode("utf-8"), digest_size=16).digest()
        h1 = int.from_bytes(h[:8], "little")
        h2 = int.from_bytes(h[8:], "little")
        for i in range(self.k):
            yield (h1 + i * h2) % self.size

    def add_check(self, item):
        """Returns True if `item` was already seen (i.e. it's a duplicate)."""
        seen = True
        for idx in self._positions(item):
            byte, bit = idx // 8, idx % 8
            if not (self.bits[byte] >> bit) & 1:
                seen = False
                self.bits[byte] |= 1 << bit
        return seen


class ShardWriter:
    def __init__(self, out_dir, prefix):
        self.out_dir = out_dir
        self.prefix = prefix
        os.makedirs(out_dir, exist_ok=True)
        self.shard_idx = 0
        self.bytes_in_shard = 0
        self.total_bytes = 0
        self._open_new_shard()

    def _open_new_shard(self):
        path = os.path.join(self.out_dir, f"{self.prefix}_{self.shard_idx:05d}.txt")
        self.f = open(path, "w", encoding="utf-8")

    def write(self, text):
        line = text + "\n"
        b = len(line.encode("utf-8"))
        if self.bytes_in_shard + b > SHARD_BYTES:
            self.f.close()
            self.shard_idx += 1
            self.bytes_in_shard = 0
            self._open_new_shard()
        self.f.write(line)
        self.bytes_in_shard += b
        self.total_bytes += b

    def close(self):
        self.f.close()


def stream_source(name, config, text_field="text"):
    ds = load_dataset(name, config, split="train", streaming=True)
    for ex in ds:
        yield ex[text_field]


def bounded(raw_iter, writer, budget_bytes):
    for item in raw_iter:
        if writer.total_bytes >= budget_bytes:
            return
        yield item


def process_stream(
    raw_iter,
    train_writer,
    val_writer,
    bloom,
    pool,
    budget_bytes,
    worker_fn,
    val_every=200,
):
    n_kept, n_seen, n_docs = 0, 0, 0

    pbar = tqdm(
        total=budget_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="⏳ Preparing Data",
        dynamic_ncols=True,
    )
    pbar.update(train_writer.total_bytes)
    last_bytes = train_writer.total_bytes

    for chunks in pool.imap_unordered(worker_fn, raw_iter, chunksize=256):
        n_docs += 1
        for cleaned in chunks:
            n_seen += 1
            if bloom.add_check(cleaned):
                continue
            n_kept += 1
            (val_writer if n_kept % val_every == 0 else train_writer).write(cleaned)

            current_bytes = train_writer.total_bytes
            bytes_added = current_bytes - last_bytes
            if bytes_added > 0:
                pbar.update(bytes_added)
                last_bytes = current_bytes

            if n_seen % 1000 == 0:
                keep_ratio = (n_kept / n_seen) * 100
                pbar.set_postfix(
                    {
                        "kept": f"{n_kept:,}",
                        "seen_chunks": f"{n_seen:,}",
                        "docs": f"{n_docs:,}",
                        "pass_rate": f"{keep_ratio:.1f}%",
                    }
                )
            if train_writer.total_bytes >= budget_bytes:
                break
        if train_writer.total_bytes >= budget_bytes:
            break
    pbar.close()
    return n_kept, n_seen


def main(args):
    bloom = BloomFilter(capacity=args.dedup_capacity)
    train_writer = ShardWriter(os.path.join(args.out_dir, "train"), "shard")
    val_writer = ShardWriter(os.path.join(args.out_dir, "val"), "shard")
    budget_bytes = int(args.target_size_gb * 1024**3)
    worker_fn = functools.partial(clean_and_chunk, max_chunk_words=args.max_chunk_words)

    with Pool(processes=args.workers) as pool:
        print("== Arabic Wikipedia ==")
        process_stream(
            bounded(
                stream_source("wikimedia/wikipedia", "20231101.ar"),
                train_writer,
                budget_bytes,
            ),
            train_writer,
            val_writer,
            bloom,
            pool,
            budget_bytes,
            worker_fn,
        )

        if train_writer.total_bytes < budget_bytes:
            print(
                "== CulturaX (ar) == (needs HF terms accepted + `huggingface-cli login`)"
            )
            process_stream(
                bounded(
                    stream_source("uonlp/CulturaX", "ar"), train_writer, budget_bytes
                ),
                train_writer,
                val_writer,
                bloom,
                pool,
                budget_bytes,
                worker_fn,
            )

    train_writer.close()
    val_writer.close()
    print(
        f"done. train ~{train_writer.total_bytes / 1024**3:.2f}GB, "
        f"val ~{val_writer.total_bytes / 1024**3:.2f}GB"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="outputs/data")
    p.add_argument(
        "--target_size_gb",
        type=float,
        default=120.0,
        help="stop once this much cleaned train text has been written",
    )
    p.add_argument(
        "--dedup_capacity",
        type=int,
        default=200_000_000,
        help="expected max number of unique docs, sizes the Bloom filter",
    )
    p.add_argument("--workers", type=int, default=64)
    p.add_argument(
        "--max_chunk_words",
        type=int,
        default=MAX_CHUNK_WORDS,
        help="split documents into chunks of at most this many words before writing",
    )
    main(p.parse_args())

```


## src/models/mamba.py

```py
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
        x = self.word_emb(input_ids)
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

```


## src/train.py

```py
"""
DDP MLM pretraining for the Bi-Mamba Arabic encoder, tuned for 4x RTX 2080Ti.

Launch:
  torchrun --nproc_per_node=4 src/train.py --config configs/base.yaml
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import datetime
import glob
import os
import pickle
import time

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from models.mamba import MambaForMaskedLM
from utils.eval import evaluate


class ShardedTextDataset(Dataset):
    """Reads lines from many shard_*.txt files without loading the corpus
    into RAM. Builds a one-time (shard_id, byte_offset) index per line,
    cached to `<shard_dir>/.index.pkl` so it's only built once."""

    def __init__(self, shard_dir, tokenizer, max_len):
        self.shard_files = sorted(glob.glob(os.path.join(shard_dir, "*.txt")))
        if not self.shard_files:
            raise FileNotFoundError(f"no shard_*.txt files found in {shard_dir}")
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.index = self._build_or_load_index(shard_dir)
        self._fh_cache = {}

    def _build_or_load_index(self, shard_dir):
        idx_path = os.path.join(shard_dir, ".index.pkl")
        if os.path.exists(idx_path):
            with open(idx_path, "rb") as f:
                return pickle.load(f)
        print(f"building line index for {shard_dir} (one-time)...")
        index = []
        for shard_id, path in enumerate(self.shard_files):
            with open(path, "rb") as f:
                offset = f.tell()
                for line in f:
                    if line.strip():
                        index.append((shard_id, offset))
                    offset = f.tell()
        with open(idx_path, "wb") as f:
            pickle.dump(index, f)
        print(f"indexed {len(index)} lines")
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        shard_id, offset = self.index[i]
        fh = self._fh_cache.get(shard_id)
        if fh is None:
            fh = open(self.shard_files[shard_id], "rb")
            self._fh_cache[shard_id] = fh
        fh.seek(offset)
        line = fh.readline().decode("utf-8").strip()
        ids = self.tokenizer.encode(
            line, add_special_tokens=True, truncation=True, max_length=self.max_len
        )
        return torch.tensor(ids, dtype=torch.long)


def collate_mlm(batch, pad_id, mask_id, vocab_size, mlm_prob):
    max_len = max(x.size(0) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((len(batch), max_len), dtype=torch.float)
    for i, seq in enumerate(batch):
        input_ids[i, : seq.size(0)] = seq
        attn_mask[i, : seq.size(0)] = 1.0
    labels = input_ids.clone()
    prob_matrix = torch.full(labels.shape, mlm_prob)
    prob_matrix[attn_mask == 0] = 0.0
    masked = torch.bernoulli(prob_matrix).bool()
    labels[~masked] = -100
    replace_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked
    input_ids[replace_mask] = mask_id
    random_mask = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked & ~replace_mask
    )
    random_tokens = torch.randint(0, vocab_size, labels.shape, dtype=torch.long)
    input_ids[random_mask] = random_tokens[random_mask]
    return input_ids, attn_mask, labels


def setup_ddp():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main = local_rank == 0

    tokenizer = AutoTokenizer.from_pretrained(cfg["data"]["tokenizer_name"])

    mcfg = dict(cfg["model"])
    mcfg["vocab_size"] = tokenizer.vocab_size
    mcfg["pad_token_id"] = tokenizer.pad_token_id
    model = MambaForMaskedLM(mcfg).to(device)
    model = DDP(model, device_ids=[local_rank])

    tcfg = cfg["training"]
    train_ds = ShardedTextDataset(
        cfg["data"]["train_dir"], tokenizer, cfg["data"]["max_seq_length"]
    )
    sampler = DistributedSampler(train_ds)
    loader = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        sampler=sampler,
        num_workers=tcfg["num_workers"],
        collate_fn=lambda b: collate_mlm(
            b,
            tokenizer.pad_token_id,
            tokenizer.mask_token_id,
            tokenizer.vocab_size,
            tcfg["mlm_probability"],
        ),
    )

    # Val set: only rank 0 evaluates (no_grad forward, no backward/allreduce
    # needed), so no DistributedSampler required here.
    val_ds = ShardedTextDataset(
        cfg["data"]["val_dir"], tokenizer, cfg["data"]["max_seq_length"]
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        num_workers=tcfg["num_workers"],
        collate_fn=lambda b: collate_mlm(
            b,
            tokenizer.pad_token_id,
            tokenizer.mask_token_id,
            tokenizer.vocab_size,
            tcfg["mlm_probability"],
        ),
    )

    optim = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=tcfg["fp16"])
    total_steps = tcfg["max_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda step: min(1.0, step / max(1, tcfg["warmup_steps"]))
        * max(0.0, (total_steps - step) / max(1, total_steps - tcfg["warmup_steps"])),
    )

    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    step = 0
    opt_step = 0
    latest_path = f"{ckpt_dir}/latest.pt"
    if os.path.exists(latest_path):
        # map_location: every rank loads onto its own GPU, not rank 0's
        ckpt = torch.load(latest_path, map_location=device)
        model.module.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        scaler.load_state_dict(ckpt["scaler"])
        scheduler.load_state_dict(ckpt["scheduler"])
        opt_step = ckpt["step"]

        if is_main:
            print(f"resumed from {latest_path} at opt_step {opt_step}")

    # --- Performance Metrics Initialization ---
    world_size = dist.get_world_size()
    effective_batch_size = tcfg["batch_size"] * world_size * tcfg["grad_accum_steps"]

    if is_main:
        print("=" * 70)
        print(f"🚀 Launching Bi-Mamba MLM Pretraining")
        print(f"   GPUs: {world_size}x RTX 2080Ti")
        print(f"   Effective Batch Size: {effective_batch_size}")
        print(f"   Total Target Steps: {total_steps}")
        print("=" * 70)

    # Trackers for execution speed, data-loading lag, and loss smoothing
    start_time = time.time()
    step_start_time = time.time()
    data_start_time = time.time()

    avg_loss = 0.0
    avg_data_time = 0.0
    avg_step_time = 0.0

    model.train()
    while opt_step < total_steps:
        sampler.set_epoch(opt_step)

        # Wrap loader in tqdm (GPU 0 only)
        pbar = tqdm(
            loader,
            desc=f"⚙️ Step {opt_step}/{total_steps}",
            disable=not is_main,
            dynamic_ncols=True,
        )

        for input_ids, attn_mask, labels in pbar:
            # Measure how long GPU was waiting on the DataLoader
            data_time = time.time() - data_start_time
            avg_data_time = 0.9 * avg_data_time + 0.1 * data_time

            input_ids, attn_mask, labels = (
                input_ids.to(device),
                attn_mask.to(device),
                labels.to(device),
            )

            with torch.amp.autocast(device_type="cuda", enabled=tcfg["fp16"]):
                out = model(input_ids, attn_mask, labels)
                loss = out["loss"] / tcfg["grad_accum_steps"]
            scaler.scale(loss).backward()
            if (step + 1) % tcfg["grad_accum_steps"] == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                scheduler.step()
                optim.zero_grad()
                opt_step += 1

                # Compute running step time and throughput
                step_time = time.time() - step_start_time
                avg_step_time = 0.9 * avg_step_time + 0.1 * step_time

                current_loss = out["loss"].item()
                avg_loss = (
                    current_loss
                    if avg_loss == 0.0
                    else 0.9 * avg_loss + 0.1 * current_loss
                )

                tokens_processed = (
                    input_ids.numel() * world_size * tcfg["grad_accum_steps"]
                )
                throughput = tokens_processed / max(0.001, step_time)

                # Update tqdm bar dynamically instead of printing a new line
                if is_main and opt_step % tcfg["log_steps"] == 0:
                    curr_lr = scheduler.get_last_lr()[0]
                    max_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

                    steps_remaining = total_steps - opt_step
                    eta_seconds = avg_step_time * steps_remaining
                    eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                    elapsed_str = str(
                        datetime.timedelta(seconds=int(time.time() - start_time))
                    )

                    pbar.set_description(f"⚙️ Step {opt_step}/{total_steps}")
                    pbar.set_postfix(
                        {
                            "Loss": f"{avg_loss:.4f}",
                            "LR": f"{curr_lr:.1e}",
                            "VRAM": f"{max_vram_gb:.1f}G",
                            "Speed": f"{throughput:.1f}t/s",
                            "DataWait": f"{avg_data_time * 1000:.0f}ms",
                            "ETA": eta_str,
                        }
                    )
                    torch.cuda.reset_peak_memory_stats(device)

                if opt_step % tcfg["save_steps"] == 0 and opt_step > 0:
                    metrics = evaluate(
                        model, val_loader, device, tcfg["fp16"], max_batches=300
                    )
                    if is_main:
                        print(
                            f"[opt_step {opt_step}] val_loss={metrics['loss']:.4f} "
                            f"val_acc={metrics['accuracy']:.4f} "
                            f"val_ppl={metrics['perplexity']:.2f}"
                        )

                        ckpt = {
                            "step": opt_step,
                            "model": model.module.state_dict(),
                            "optim": optim.state_dict(),
                            "scaler": scaler.state_dict(),
                            "scheduler": scheduler.state_dict(),
                        }
                        ckpt_path = (
                            f"{cfg['paths']['checkpoint_dir']}/step_{opt_step}.pt"
                        )
                        torch.save(ckpt, ckpt_path)
                        torch.save(ckpt, f"{cfg['paths']['checkpoint_dir']}/latest.pt")
                step_start_time = time.time()
            step += 1
            data_start_time = time.time()
            if opt_step >= total_steps:
                break

    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    args = p.parse_args()
    main(args.config)

```


## src/utils/eval.py

```py
"""
Held-out evaluation for MLM pretraining: val loss, masked-token accuracy,
and perplexity. Kept separate from train.py so the eval logic can be run,
tested, or extended (e.g. later a downstream probe) independently of the
training loop.
"""

import math

import torch


@torch.no_grad()
def evaluate(model, val_loader, device, fp16, max_batches=None):
    """Runs a full pass over val_loader and returns a dict of metrics:
      - loss: mean MLM cross-entropy loss over all masked positions
      - accuracy: top-1 accuracy of predictions at masked positions only
      - perplexity: exp(loss), the standard MLM perplexity metric

    Only meant to be called on the main rank (rank 0); this is a no_grad
    forward pass with no backward/allreduce, so DDP doesn't need every
    rank to participate.
    """
    model.eval()
    total_loss, n_batches = 0.0, 0
    correct, total_masked = 0, 0

    for i, (input_ids, attn_mask, labels) in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        input_ids, attn_mask, labels = (
            input_ids.to(device),
            attn_mask.to(device),
            labels.to(device),
        )
        with torch.amp.autocast(device_type="cuda", enabled=fp16):
            out = model(input_ids, attn_mask, labels)

        total_loss += out["loss"].item()
        n_batches += 1

        masked_positions = labels != -100
        if masked_positions.any():
            preds = out["logits"].argmax(dim=-1)
            correct += (
                (preds[masked_positions] == labels[masked_positions]).sum().item()
            )
            total_masked += masked_positions.sum().item()

    model.train()

    mean_loss = total_loss / max(1, n_batches)
    accuracy = correct / max(1, total_masked)
    # guard against overflow on a badly-diverged run
    perplexity = math.exp(mean_loss) if mean_loss < 20 else float("inf")

    return {"loss": mean_loss, "accuracy": accuracy, "perplexity": perplexity}

```

