# Tree View:
```
.
├── configs
│   ├── base.yaml
│   ├── dry_test.yaml
│   ├── eval_check.yaml
│   ├── finetune_anercorp.yaml
│   ├── finetune_anercorp_seed1337.yaml
│   ├── finetune_anercorp_seed2024.yaml
│   ├── finetune_anercorp_seed42.yaml
│   ├── finetune_arcd.yaml
│   ├── finetune_arcd_seed1337.yaml
│   ├── finetune_arcd_seed2024.yaml
│   ├── finetune_arcd_seed42.yaml
│   ├── finetune_hard.yaml
│   ├── finetune_hard_seed1337.yaml
│   ├── finetune_hard_seed2024.yaml
│   ├── finetune_hard_seed42.yaml
│   ├── finetune_squad_plus_arcd50.yaml
│   ├── finetune_xnli.yaml
│   ├── finetune_xnli_seed1337.yaml
│   ├── finetune_xnli_seed2024.yaml
│   ├── finetune_xnli_seed42.yaml
│   ├── pilot.yaml
│   ├── prefinetune_anercorp.yaml
│   ├── prefinetune_arcd.yaml
│   └── prefinetune_arcd_v2.yaml
├── src
│   ├── data
│   │   ├── finetune_datasets.py
│   │   └── prepare_data.py
│   ├── finetune.py
│   ├── models
│   │   ├── mamba-old.py
│   │   ├── mamba.py
│   │   └── mamba_heads.py
│   ├── sanity
│   │   └── test.py
│   ├── train.py
│   └── utils
│       ├── aggregate_results.py
│       ├── eval.py
│       └── finetune_metrics.py
└── test.py

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
  gradient_checkpointing: true
training:
  batch_size: 8
  grad_accum_steps: 8
  lr: 3.0e-4
  weight_decay: 0.01
  warmup_steps: 10000
  max_steps: 200000
  mlm_probability: 0.15
  fp16: true
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
  gradient_checkpointing: true
training:
  batch_size: 8
  grad_accum_steps: 8
  lr: 3.0e-4
  weight_decay: 0.01
  warmup_steps: 10000
  max_steps: 60
  mlm_probability: 0.15
  fp16: true
  seed: 42
  save_steps: 10
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


## configs/finetune_anercorp.yaml

```yaml
task:
  name: "anercorp"
# checkpoint if you haven't run the prefinetune step yet.
pretrained_ckpt: "results_finetune/prefinetune_anercorp/polyglot_ner_ar_best.pt"
model:
  d_model: 512
  n_layer: 12
  d_state: 16
  d_conv: 4
  expand: 2
  max_position_embeddings: 512
  dropout: 0.1
  gradient_checkpointing: false
training:
  batch_size: 4
  eval_batch_size: 32
  lr: 2.0e-5
  weight_decay: 0.01
  warmup_steps: 50
  max_grad_norm: 1.0
  epochs: 5
  fp16: true
  seed: 42
  num_workers: 4
data:
  tokenizer_name: "aubmindlab/bert-base-arabertv02"
  max_seq_length: 128
paths:
  output_dir: "results_finetune/anercorp"
```


## configs/finetune_anercorp_seed1337.yaml

```yaml
data:
  max_seq_length: 128
  tokenizer_name: aubmindlab/bert-base-arabertv02
model:
  d_conv: 4
  d_model: 512
  d_state: 16
  dropout: 0.1
  expand: 2
  gradient_checkpointing: false
  max_position_embeddings: 512
  n_layer: 12
paths:
  output_dir: results_finetune/anercorp
pretrained_ckpt: results_finetune/prefinetune_anercorp/polyglot_ner_ar_best.pt
task:
  name: anercorp
training:
  batch_size: 4
  epochs: 5
  eval_batch_size: 32
  fp16: true
  lr: 2.0e-05
  max_grad_norm: 1.0
  num_workers: 4
  seed: 1337
  warmup_steps: 50
  weight_decay: 0.01
```


## configs/finetune_anercorp_seed2024.yaml

```yaml
data:
  max_seq_length: 128
  tokenizer_name: aubmindlab/bert-base-arabertv02
model:
  d_conv: 4
  d_model: 512
  d_state: 16
  dropout: 0.1
  expand: 2
  gradient_checkpointing: false
  max_position_embeddings: 512
  n_layer: 12
paths:
  output_dir: results_finetune/anercorp
pretrained_ckpt: results_finetune/prefinetune_anercorp/polyglot_ner_ar_best.pt
task:
  name: anercorp
training:
  batch_size: 4
  epochs: 5
  eval_batch_size: 32
  fp16: true
  lr: 2.0e-05
  max_grad_norm: 1.0
  num_workers: 4
  seed: 2024
  warmup_steps: 50
  weight_decay: 0.01
```


## configs/finetune_anercorp_seed42.yaml

```yaml
data:
  max_seq_length: 128
  tokenizer_name: aubmindlab/bert-base-arabertv02
model:
  d_conv: 4
  d_model: 512
  d_state: 16
  dropout: 0.1
  expand: 2
  gradient_checkpointing: false
  max_position_embeddings: 512
  n_layer: 12
paths:
  output_dir: results_finetune/anercorp
pretrained_ckpt: results_finetune/prefinetune_anercorp/polyglot_ner_ar_best.pt
task:
  name: anercorp
training:
  batch_size: 4
  epochs: 5
  eval_batch_size: 32
  fp16: true
  lr: 2.0e-05
  max_grad_norm: 1.0
  num_workers: 4
  seed: 42
  warmup_steps: 50
  weight_decay: 0.01
```


## configs/finetune_arcd.yaml

```yaml
task:
  name: "arcd"
pretrained_ckpt: "results_finetune/prefinetune_arcd_v2/squad_plus_tydiqa_ar_best.pt"
model:
  d_model: 512
  n_layer: 12
  d_state: 16
  d_conv: 4
  expand: 2
  max_position_embeddings: 512
  dropout: 0.1
  gradient_checkpointing: false
training:
  batch_size: 8
  eval_batch_size: 16
  lr: 3.0e-5
  weight_decay: 0.01
  epochs: 10
  warmup_steps: 50
  fp16: true
  seed: 42
  num_workers: 4
data:
  tokenizer_name: "aubmindlab/bert-base-arabertv02"
  max_seq_length: 384
paths:
  output_dir: "results_finetune/arcd"
```


## configs/finetune_arcd_seed1337.yaml

```yaml
data:
  max_seq_length: 384
  tokenizer_name: aubmindlab/bert-base-arabertv02
model:
  d_conv: 4
  d_model: 512
  d_state: 16
  dropout: 0.1
  expand: 2
  gradient_checkpointing: false
  max_position_embeddings: 512
  n_layer: 12
paths:
  output_dir: results_finetune/arcd
pretrained_ckpt: results_finetune/prefinetune_arcd_v2/squad_plus_tydiqa_ar_best.pt
task:
  name: arcd
training:
  batch_size: 8
  epochs: 10
  eval_batch_size: 16
  fp16: true
  lr: 3.0e-05
  num_workers: 4
  seed: 1337
  warmup_steps: 50
  weight_decay: 0.01
```


## configs/finetune_arcd_seed2024.yaml

```yaml
data:
  max_seq_length: 384
  tokenizer_name: aubmindlab/bert-base-arabertv02
model:
  d_conv: 4
  d_model: 512
  d_state: 16
  dropout: 0.1
  expand: 2
  gradient_checkpointing: false
  max_position_embeddings: 512
  n_layer: 12
paths:
  output_dir: results_finetune/arcd
pretrained_ckpt: results_finetune/prefinetune_arcd_v2/squad_plus_tydiqa_ar_best.pt
task:
  name: arcd
training:
  batch_size: 8
  epochs: 10
  eval_batch_size: 16
  fp16: true
  lr: 3.0e-05
  num_workers: 4
  seed: 2024
  warmup_steps: 50
  weight_decay: 0.01
```


## configs/finetune_arcd_seed42.yaml

```yaml
data:
  max_seq_length: 384
  tokenizer_name: aubmindlab/bert-base-arabertv02
model:
  d_conv: 4
  d_model: 512
  d_state: 16
  dropout: 0.1
  expand: 2
  gradient_checkpointing: false
  max_position_embeddings: 512
  n_layer: 12
paths:
  output_dir: results_finetune/arcd
pretrained_ckpt: results_finetune/prefinetune_arcd_v2/squad_plus_tydiqa_ar_best.pt
task:
  name: arcd
training:
  batch_size: 8
  epochs: 10
  eval_batch_size: 16
  fp16: true
  lr: 3.0e-05
  num_workers: 4
  seed: 42
  warmup_steps: 50
  weight_decay: 0.01
```


## configs/finetune_hard.yaml

```yaml
task:
  name: "hard"        # sentiment, HARD hotel reviews, MSA-leaning but not pure MSA
pretrained_ckpt: "checkpoint/latest.pt"   # your 200k-step base.yaml checkpoint
model:
  d_model: 512
  n_layer: 12
  d_state: 16
  d_conv: 4
  expand: 2
  max_position_embeddings: 512
  dropout: 0.1
  gradient_checkpointing: false
training:
  batch_size: 16
  eval_batch_size: 32
  lr: 2.0e-5
  weight_decay: 0.01
  epochs: 4
  fp16: true
  seed: 42
  num_workers: 4
data:
  tokenizer_name: "aubmindlab/bert-base-arabertv02"
  max_seq_length: 256
paths:
  output_dir: "results_finetune/hard"
```


## configs/finetune_hard_seed1337.yaml

```yaml
data:
  max_seq_length: 256
  tokenizer_name: aubmindlab/bert-base-arabertv02
model:
  d_conv: 4
  d_model: 512
  d_state: 16
  dropout: 0.1
  expand: 2
  gradient_checkpointing: false
  max_position_embeddings: 512
  n_layer: 12
paths:
  output_dir: results_finetune/hard
pretrained_ckpt: checkpoint/latest.pt
task:
  name: hard
training:
  batch_size: 16
  epochs: 4
  eval_batch_size: 32
  fp16: true
  lr: 2.0e-05
  num_workers: 4
  seed: 1337
  weight_decay: 0.01
```


## configs/finetune_hard_seed2024.yaml

```yaml
data:
  max_seq_length: 256
  tokenizer_name: aubmindlab/bert-base-arabertv02
model:
  d_conv: 4
  d_model: 512
  d_state: 16
  dropout: 0.1
  expand: 2
  gradient_checkpointing: false
  max_position_embeddings: 512
  n_layer: 12
paths:
  output_dir: results_finetune/hard
pretrained_ckpt: checkpoint/latest.pt
task:
  name: hard
training:
  batch_size: 16
  epochs: 4
  eval_batch_size: 32
  fp16: true
  lr: 2.0e-05
  num_workers: 4
  seed: 2024
  weight_decay: 0.01
```


## configs/finetune_hard_seed42.yaml

```yaml
data:
  max_seq_length: 256
  tokenizer_name: aubmindlab/bert-base-arabertv02
model:
  d_conv: 4
  d_model: 512
  d_state: 16
  dropout: 0.1
  expand: 2
  gradient_checkpointing: false
  max_position_embeddings: 512
  n_layer: 12
paths:
  output_dir: results_finetune/hard
pretrained_ckpt: checkpoint/latest.pt
task:
  name: hard
training:
  batch_size: 16
  epochs: 4
  eval_batch_size: 32
  fp16: true
  lr: 2.0e-05
  num_workers: 4
  seed: 42
  weight_decay: 0.01
```


## configs/finetune_squad_plus_arcd50.yaml

```yaml
task:
  name: "squad_plus_arcd50"
pretrained_ckpt: "checkpoint/latest.pt"
model:
  d_model: 512
  n_layer: 12
  d_state: 16
  d_conv: 4
  expand: 2
  max_position_embeddings: 512
  dropout: 0.1
  gradient_checkpointing: false
training:
  batch_size: 8
  eval_batch_size: 16
  lr: 3.0e-5
  weight_decay: 0.01
  epochs: 5
  warmup_steps: 500
  fp16: true
  seed: 42
  num_workers: 4
data:
  tokenizer_name: "aubmindlab/bert-base-arabertv02"
  max_seq_length: 512
paths:
  output_dir: "results_finetune/squad_plus_arcd50"
```


## configs/finetune_xnli.yaml

```yaml
task:
  name: "xnli_ar"
pretrained_ckpt: "checkpoint/latest.pt"
model:
  d_model: 512
  n_layer: 12
  d_state: 16
  d_conv: 4
  expand: 2
  max_position_embeddings: 512
  dropout: 0.1
  gradient_checkpointing: false
training:
  batch_size: 16
  eval_batch_size: 32
  lr: 2.0e-5
  weight_decay: 0.01
  epochs: 3
  fp16: true
  seed: 42
  num_workers: 4
data:
  tokenizer_name: "aubmindlab/bert-base-arabertv02"
  max_seq_length: 128
paths:
  output_dir: "results_finetune/xnli_ar"
```


## configs/finetune_xnli_seed1337.yaml

```yaml
data:
  max_seq_length: 128
  tokenizer_name: aubmindlab/bert-base-arabertv02
model:
  d_conv: 4
  d_model: 512
  d_state: 16
  dropout: 0.1
  expand: 2
  gradient_checkpointing: false
  max_position_embeddings: 512
  n_layer: 12
paths:
  output_dir: results_finetune/xnli_ar
pretrained_ckpt: checkpoint/latest.pt
task:
  name: xnli_ar
training:
  batch_size: 16
  epochs: 3
  eval_batch_size: 32
  fp16: true
  lr: 2.0e-05
  num_workers: 4
  seed: 1337
  weight_decay: 0.01
```


## configs/finetune_xnli_seed2024.yaml

```yaml
data:
  max_seq_length: 128
  tokenizer_name: aubmindlab/bert-base-arabertv02
model:
  d_conv: 4
  d_model: 512
  d_state: 16
  dropout: 0.1
  expand: 2
  gradient_checkpointing: false
  max_position_embeddings: 512
  n_layer: 12
paths:
  output_dir: results_finetune/xnli_ar
pretrained_ckpt: checkpoint/latest.pt
task:
  name: xnli_ar
training:
  batch_size: 16
  epochs: 3
  eval_batch_size: 32
  fp16: true
  lr: 2.0e-05
  num_workers: 4
  seed: 2024
  weight_decay: 0.01
```


## configs/finetune_xnli_seed42.yaml

```yaml
data:
  max_seq_length: 128
  tokenizer_name: aubmindlab/bert-base-arabertv02
model:
  d_conv: 4
  d_model: 512
  d_state: 16
  dropout: 0.1
  expand: 2
  gradient_checkpointing: false
  max_position_embeddings: 512
  n_layer: 12
paths:
  output_dir: results_finetune/xnli_ar
pretrained_ckpt: checkpoint/latest.pt
task:
  name: xnli_ar
training:
  batch_size: 16
  epochs: 3
  eval_batch_size: 32
  fp16: true
  lr: 2.0e-05
  num_workers: 4
  seed: 42
  weight_decay: 0.01
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
  gradient_checkpointing: true
training:
  batch_size: 8
  grad_accum_steps: 8
  lr: 5.0e-4
  weight_decay: 0.01
  warmup_steps: 200
  max_steps: 2000
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


## configs/prefinetune_anercorp.yaml

```yaml
# Polyglot-NER's Arabic config (silver Wikipedia NER, capped at 150k
# finetune_anercorp.yaml's pretrained_ckpt at this run's output best.pt
# once it's done:
#   pretrained_ckpt: "results_finetune/prefinetune_anercorp/polyglot_ner_ar_best.pt"
# NOTE: Polyglot-NER's tagset (O/PER/LOC/ORG, not BIO-prefixed) differs
# from ANERcorp's (BIO + MISC) -- that's fine, only the encoder weights
# carry over (load_pretrained_encoder only pulls "encoder.*" keys), the
# ANERcorp's own tagset.
task:
  name: "polyglot_ner_ar"
pretrained_ckpt: "checkpoint/latest.pt"
model:
  d_model: 512
  n_layer: 12
  d_state: 16
  d_conv: 4
  expand: 2
  max_position_embeddings: 512
  dropout: 0.1
  gradient_checkpointing: false
training:
  batch_size: 16
  eval_batch_size: 32
  lr: 3.0e-5
  weight_decay: 0.01
  epochs: 2
  warmup_steps: 200
  fp16: true
  seed: 42
  num_workers: 4
data:
  tokenizer_name: "aubmindlab/bert-base-arabertv02"
  max_seq_length: 128
paths:
  output_dir: "results_finetune/prefinetune_anercorp"
```


## configs/prefinetune_arcd.yaml

```yaml
# (~700 train example) ARCD gold set. Point finetune_arcd.yaml's
# pretrained_ckpt at this run's output best.pt once it's done:
#   pretrained_ckpt: "results_finetune/prefinetune_arcd/arabic_squad_best.pt"
task:
  name: "arabic_squad"
pretrained_ckpt: "checkpoint/latest.pt"
model:
  d_model: 512
  n_layer: 12
  d_state: 16
  d_conv: 4
  expand: 2
  max_position_embeddings: 512
  dropout: 0.1
  gradient_checkpointing: false
training:
  batch_size: 16          # bigger corpus, less overfitting risk than ARCD's 8
  eval_batch_size: 32
  lr: 3.0e-5
  weight_decay: 0.01
  epochs: 2
  warmup_steps: 200
  fp16: true
  seed: 42
  num_workers: 4
data:
  tokenizer_name: "aubmindlab/bert-base-arabertv02"
  max_seq_length: 384     # SQuAD paragraphs are generally shorter than ARCD's
paths:
  output_dir: "results_finetune/prefinetune_arcd"
```


## configs/prefinetune_arcd_v2.yaml

```yaml
task:
  name: "squad_plus_tydiqa_ar"
pretrained_ckpt: "checkpoint/latest.pt"
model:
  d_model: 512
  n_layer: 12
  d_state: 16
  d_conv: 4
  expand: 2
  max_position_embeddings: 512
  dropout: 0.1
  gradient_checkpointing: false
training:
  batch_size: 16
  eval_batch_size: 32
  lr: 3.0e-5
  weight_decay: 0.01
  epochs: 2
  warmup_steps: 200
  fp16: true
  seed: 42
  num_workers: 4
data:
  tokenizer_name: "aubmindlab/bert-base-arabertv02"
  max_seq_length: 384
paths:
  output_dir: "results_finetune/prefinetune_arcd_v2"
```


## src/data/finetune_datasets.py

```py
import random
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            texts, truncation=True, max_length=max_len, padding=False
        )["input_ids"]
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.encodings[i], dtype=torch.long),
            "label": torch.tensor(self.labels[i], dtype=torch.long),
        }
def collate_classification(batch, pad_id):
    max_len = max(x["input_ids"].size(0) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((len(batch), max_len), dtype=torch.float)
    labels = torch.stack([x["label"] for x in batch])
    for i, x in enumerate(batch):
        L = x["input_ids"].size(0)
        input_ids[i, :L] = x["input_ids"]
        attn_mask[i, :L] = 1.0
    return input_ids, attn_mask, labels
def _split_train_only(texts, labels, seed=42, train_frac=0.8, dev_frac=0.1):
    n = len(texts)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_train = int(n * train_frac)
    n_dev = int(n * dev_frac)
    def subset(lo, hi):
        return [texts[i] for i in idx[lo:hi]], [labels[i] for i in idx[lo:hi]]
    return (
        subset(0, n_train),
        subset(n_train, n_train + n_dev),
        subset(n_train + n_dev, n),
    )
def load_hard(tokenizer, max_len):
    ds = load_dataset("Elnagara/hard")
    texts, labels = [], []
    for ex in ds["train"]:
        label_idx = ex["label"]
        if label_idx == 2:
            continue
        texts.append(ex["text"])
        labels.append(1 if label_idx > 2 else 0)
    (tr_t, tr_l), (dv_t, dv_l), (te_t, te_l) = _split_train_only(texts, labels)
    return (
        ClassificationDataset(tr_t, tr_l, tokenizer, max_len),
        ClassificationDataset(dv_t, dv_l, tokenizer, max_len),
        ClassificationDataset(te_t, te_l, tokenizer, max_len),
    )
def load_xnli_ar(tokenizer, max_len):
    ds = load_dataset("facebook/xnli", "ar")
    def build(split):
        premises = [str(p) for p in ds[split]["premise"]]
        hyps = [str(h) for h in ds[split]["hypothesis"]]
        labels = [int(l) for l in ds[split]["label"]]
        enc = tokenizer(
            premises, hyps, truncation=True, max_length=max_len, padding=False
        )
        cds = ClassificationDataset.__new__(ClassificationDataset)
        cds.encodings = enc["input_ids"]
        cds.labels = labels
        return cds
    return build("train"), build("validation"), build("test")
class TokenClassificationDataset(Dataset):
    def __init__(self, word_lists, tag_lists, tokenizer, max_len, label2id):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
        self.word_lists = word_lists
        self.tag_lists = tag_lists
    def __len__(self):
        return len(self.word_lists)
    def __getitem__(self, i):
        words = self.word_lists[i]
        tags = self.tag_lists[i]
        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            return_offsets_mapping=False,
        )
        word_ids = enc.word_ids()
        labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            elif wid != prev_word_id:
                labels.append(self.label2id[tags[wid]])
            else:
                labels.append(-100)
            prev_word_id = wid
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
def collate_ner(batch, pad_id):
    max_len = max(x["input_ids"].size(0) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((len(batch), max_len), dtype=torch.float)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, x in enumerate(batch):
        L = x["input_ids"].size(0)
        input_ids[i, :L] = x["input_ids"]
        attn_mask[i, :L] = 1.0
        labels[i, :L] = x["labels"]
    return input_ids, attn_mask, labels
SENTENCE_END_TOKENS = {".", "؟", "!"}
def _group_into_pseudo_sentences(words, tags, max_words=120):
    sentences_words, sentences_tags = [], []
    cur_w, cur_t = [], []
    for w, t in zip(words, tags):
        cur_w.append(w)
        cur_t.append(t)
        ends_sentence = w in SENTENCE_END_TOKENS and t == "O"
        if ends_sentence or len(cur_w) >= max_words:
            sentences_words.append(cur_w)
            sentences_tags.append(cur_t)
            cur_w, cur_t = [], []
    if cur_w:
        sentences_words.append(cur_w)
        sentences_tags.append(cur_t)
    return sentences_words, sentences_tags
def load_anercorp(tokenizer, max_len):
    ds = load_dataset("asas-ai/ANERCorp")
    all_tags = sorted(set(ds["train"]["tag"]) | set(ds["test"]["tag"]))
    label2id = {t: i for i, t in enumerate(all_tags)}
    train_sw, train_st = _group_into_pseudo_sentences(
        ds["train"]["word"], ds["train"]["tag"]
    )
    test_sw, test_st = _group_into_pseudo_sentences(
        ds["test"]["word"], ds["test"]["tag"]
    )
    idx = list(range(len(train_sw)))
    random.Random(42).shuffle(idx)
    n_dev = max(1, int(0.1 * len(idx)))
    dev_idx = set(idx[:n_dev])
    tr_w = [train_sw[i] for i in idx if i not in dev_idx]
    tr_t = [train_st[i] for i in idx if i not in dev_idx]
    dv_w = [train_sw[i] for i in dev_idx]
    dv_t = [train_st[i] for i in dev_idx]
    train_ds = TokenClassificationDataset(tr_w, tr_t, tokenizer, max_len, label2id)
    dev_ds = TokenClassificationDataset(dv_w, dv_t, tokenizer, max_len, label2id)
    test_ds = TokenClassificationDataset(test_sw, test_st, tokenizer, max_len, label2id)
    train_ds.label2id = label2id
    return train_ds, dev_ds, test_ds
class QADataset(Dataset):
    def __init__(self, examples, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.features = []
        n_dropped = 0
        for ex in examples:
            feat = self._build_feature(ex, tokenizer, max_len)
            if feat is None:
                n_dropped += 1
                continue
            self.features.append(feat)
        if n_dropped:
            print(
                f"[QADataset] dropped {n_dropped}/{len(examples)} examples whose "
                "answer span fell outside the truncated context"
            )
    @staticmethod
    def _build_feature(ex, tokenizer, max_len):
        question = ex["question"]
        context = ex["context"]
        answer_text = ex["answers"]["text"][0]
        answer_start = ex["answers"]["answer_start"][0]
        enc = tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=max_len,
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")
        sequence_ids = enc.sequence_ids()
        ctx_start_tok = sequence_ids.index(1)
        ctx_end_tok = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)
        char_start = answer_start
        char_end = answer_start + len(answer_text)
        if not (
            offsets[ctx_start_tok][0] <= char_start
            and offsets[ctx_end_tok][1] >= char_end
        ):
            return None
        tok_start = ctx_start_tok
        while tok_start <= ctx_end_tok and offsets[tok_start][0] <= char_start:
            tok_start += 1
        tok_start -= 1
        tok_end = ctx_end_tok
        while tok_end >= ctx_start_tok and offsets[tok_end][1] >= char_end:
            tok_end -= 1
        tok_end += 1
        return {
            "id": ex["id"],
            "input_ids": enc["input_ids"],
            "start": tok_start,
            "end": tok_end,
            "gold_answers": ex["answers"]["text"],
        }
    def __len__(self):
        return len(self.features)
    def __getitem__(self, i):
        f = self.features[i]
        return {
            "id": f["id"],
            "input_ids": torch.tensor(f["input_ids"], dtype=torch.long),
            "start": torch.tensor(f["start"], dtype=torch.long),
            "end": torch.tensor(f["end"], dtype=torch.long),
            "gold_answers": f["gold_answers"],
        }
def collate_qa(batch, pad_id):
    max_len = max(x["input_ids"].size(0) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((len(batch), max_len), dtype=torch.float)
    starts = torch.stack([x["start"] for x in batch])
    ends = torch.stack([x["end"] for x in batch])
    ids = [x["id"] for x in batch]
    gold = [x["gold_answers"] for x in batch]
    for i, x in enumerate(batch):
        L = x["input_ids"].size(0)
        input_ids[i, :L] = x["input_ids"]
        attn_mask[i, :L] = 1.0
    return input_ids, attn_mask, starts, ends, ids, gold
def load_arcd(tokenizer, max_len):
    ds = load_dataset("hsseinmz/arcd")
    train_ex = list(ds["train"])
    val_ex = list(ds["validation"])
    random.Random(42).shuffle(val_ex)
    half = len(val_ex) // 2
    dev_ex, test_ex = val_ex[:half], val_ex[half:]
    return (
        QADataset(train_ex, tokenizer, max_len),
        QADataset(dev_ex, tokenizer, max_len),
        QADataset(test_ex, tokenizer, max_len),
    )
def _flatten_squad_style(raw_data):
    examples = []
    for article in raw_data:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                answers = qa.get("answers") or []
                if not answers:
                    continue
                examples.append(
                    {
                        "id": qa["id"],
                        "question": qa["question"],
                        "context": context,
                        "answers": {
                            "text": [a["text"] for a in answers],
                            "answer_start": [a["answer_start"] for a in answers],
                        },
                    }
                )
    return examples
def load_arabic_squad(tokenizer, max_len):
    ds = load_dataset("i0xs0/Arabic-SQuAD")
    raw_data = ds["train"][0]["data"]
    examples = _flatten_squad_style(raw_data)
    random.Random(42).shuffle(examples)
    n = len(examples)
    n_train = int(n * 0.90)
    n_dev = int(n * 0.05)
    train_ex = examples[:n_train]
    dev_ex = examples[n_train : n_train + n_dev]
    test_ex = examples[n_train + n_dev :]
    return (
        QADataset(train_ex, tokenizer, max_len),
        QADataset(dev_ex, tokenizer, max_len),
        QADataset(test_ex, tokenizer, max_len),
    )
def load_arcd_50_50(tokenizer, max_len, seed=42):
    ds = load_dataset("hsseinmz/arcd")
    all_ex = list(ds["train"]) + list(ds["validation"])
    random.Random(seed).shuffle(all_ex)
    half = len(all_ex) // 2
    arcd_train_pool = all_ex[:half]
    test_ex = all_ex[half:]
    n_dev = int(len(arcd_train_pool) * 0.10)
    dev_ex = arcd_train_pool[:n_dev]
    train_ex = arcd_train_pool[n_dev:]
    return (
        QADataset(train_ex, tokenizer, max_len),
        QADataset(dev_ex, tokenizer, max_len),
        QADataset(test_ex, tokenizer, max_len),
    )
def load_squad_plus_arcd50(tokenizer, max_len, seed=42):
    ds_squad = load_dataset("i0xs0/Arabic-SQuAD")
    raw_squad = ds_squad["train"][0]["data"]
    squad_examples = _flatten_squad_style(raw_squad)
    random.Random(seed).shuffle(squad_examples)
    n_squad_dev = 2000
    squad_train_ex = squad_examples[:-n_squad_dev]
    squad_dev_ex = squad_examples[-n_squad_dev:]
    ds_arcd = load_dataset("hsseinmz/arcd")
    all_arcd = list(ds_arcd["train"]) + list(ds_arcd["validation"])
    random.Random(seed).shuffle(all_arcd)
    half = len(all_arcd) // 2
    arcd_train_pool = all_arcd[:half]
    test_ex = all_arcd[half:]
    n_arcd_dev = int(len(arcd_train_pool) * 0.10)
    arcd_dev_ex = arcd_train_pool[:n_arcd_dev]
    arcd_train_ex = arcd_train_pool[n_arcd_dev:]
    joint_train_ex = squad_train_ex + arcd_train_ex
    joint_dev_ex = squad_dev_ex + arcd_dev_ex
    random.Random(seed).shuffle(joint_train_ex)
    print(
        f"[QA Loader] Joint setup: {len(joint_train_ex)} train "
        f"({len(squad_train_ex)} SQuAD + {len(arcd_train_ex)} ARCD) | "
        f"{len(joint_dev_ex)} dev | {len(test_ex)} test"
    )
    return (
        QADataset(joint_train_ex, tokenizer, max_len),
        QADataset(joint_dev_ex, tokenizer, max_len),
        QADataset(test_ex, tokenizer, max_len),
    )
def load_tydiqa_ar(tokenizer, max_len):
    ds = load_dataset("google-research-datasets/tydiqa", "secondary_task")
    def build(split):
        out = []
        for ex in ds[split]:
            if not ex["id"].startswith("arabic"):
                continue
            if not ex["answers"]["text"]:
                continue
            out.append({
                "id": ex["id"],
                "question": ex["question"],
                "context": ex["context"],
                "answers": {
                    "text": ex["answers"]["text"],
                    "answer_start": ex["answers"]["answer_start"],
                },
            })
        return out
    train_ex = build("train")
    val_ex = build("validation")
    random.Random(42).shuffle(val_ex)
    half = len(val_ex) // 2
    dev_ex, test_ex = val_ex[:half], val_ex[half:]
    return (
        QADataset(train_ex, tokenizer, max_len),
        QADataset(dev_ex, tokenizer, max_len),
        QADataset(test_ex, tokenizer, max_len),
    )
def load_squad_plus_tydiqa_ar(tokenizer, max_len, seed=42):
    ds_squad = load_dataset("i0xs0/Arabic-SQuAD")
    squad_examples = _flatten_squad_style(ds_squad["train"][0]["data"])
    ds_tydi = load_dataset("google-research-datasets/tydiqa", "secondary_task")
    def tydi_build(split):
        return [
            {
                "id": ex["id"],
                "question": ex["question"],
                "context": ex["context"],
                "answers": {
                    "text": ex["answers"]["text"],
                    "answer_start": ex["answers"]["answer_start"],
                },
            }
            for ex in ds_tydi[split]
            if ex["id"].startswith("arabic") and ex["answers"]["text"]
        ]
    tydi_train = tydi_build("train")
    tydi_val = tydi_build("validation")
    random.Random(seed).shuffle(tydi_val)
    half = len(tydi_val) // 2
    tydi_dev, tydi_test = tydi_val[:half], tydi_val[half:]
    random.Random(seed).shuffle(squad_examples)
    n_squad_dev = 2000
    squad_train = squad_examples[:-n_squad_dev]
    squad_dev = squad_examples[-n_squad_dev:]
    train_ex = squad_train + tydi_train
    random.Random(seed).shuffle(train_ex)
    dev_ex = squad_dev + tydi_dev
    print(
        f"[QA Loader] squad+tydiqa_ar: {len(train_ex)} train "
        f"({len(squad_train)} squad + {len(tydi_train)} tydiqa) | "
        f"{len(dev_ex)} dev | {len(tydi_test)} test (tydiqa-ar only)"
    )
    return (
        QADataset(train_ex, tokenizer, max_len),
        QADataset(dev_ex, tokenizer, max_len),
        QADataset(tydi_test, tokenizer, max_len),
    )
_POLYGLOT_NER_MAX_SENTENCES = 150_000
def load_polyglot_ner_ar(tokenizer, max_len):
    PREFINETUNE_SAMPLE_SIZE = _POLYGLOT_NER_MAX_SENTENCES
    ds = load_dataset(
        "rmyeid/polyglot_ner",
        "default",
        revision="refs/convert/parquet",
        streaming=True,
    )
    ar_stream = ds["train"].filter(lambda ex: ex["lang"] == "ar")
    words_col, tags_col = [], []
    for ex in ar_stream.take(PREFINETUNE_SAMPLE_SIZE):
        words_col.append(ex["words"])
        tags_col.append(ex["ner"])
    idx = list(range(len(words_col)))
    random.Random(42).shuffle(idx)
    n = len(idx)
    n_train = int(n * 0.90)
    n_dev = int(n * 0.05)
    train_idx = idx[:n_train]
    dev_idx = idx[n_train : n_train + n_dev]
    test_idx = idx[n_train + n_dev :]
    all_tags = sorted({t for i in idx for t in tags_col[i]})
    label2id = {t: i for i, t in enumerate(all_tags)}
    def subset(indices):
        return [words_col[i] for i in indices], [tags_col[i] for i in indices]
    tr_w, tr_t = subset(train_idx)
    dv_w, dv_t = subset(dev_idx)
    te_w, te_t = subset(test_idx)
    train_ds = TokenClassificationDataset(tr_w, tr_t, tokenizer, max_len, label2id)
    dev_ds = TokenClassificationDataset(dv_w, dv_t, tokenizer, max_len, label2id)
    test_ds = TokenClassificationDataset(te_w, te_t, tokenizer, max_len, label2id)
    train_ds.label2id = label2id
    return train_ds, dev_ds, test_ds
```


## src/data/prepare_data.py

```py
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
SHARD_BYTES = 1 * 1024**3
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
    cleaned = clean_doc(text)
    if cleaned is None:
        return []
    return list(chunk_doc(cleaned, max_words=max_chunk_words))
class BloomFilter:
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


## src/finetune.py

```py
import argparse
import datetime
import json
import os
import random
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from data.finetune_datasets import (
    collate_classification,
    collate_ner,
    collate_qa,
    load_anercorp,
    load_arabic_squad,
    load_arcd,
    load_arcd_50_50,
    load_hard,
    load_polyglot_ner_ar,
    load_squad_plus_arcd50,
    load_squad_plus_tydiqa_ar,
    load_xnli_ar,
)
from models.mamba_heads import (
    MambaForQuestionAnswering,
    MambaForSequenceClassification,
    MambaForTokenClassification,
    load_pretrained_encoder,
)
from utils.finetune_metrics import (
    classification_metrics,
    qa_metrics,
    seqeval_ner_metrics,
)
TASK_REGISTRY = {
    "hard": {"loader": load_hard, "task_type": "classification", "num_labels": 2},
    "xnli_ar": {"loader": load_xnli_ar, "task_type": "classification", "num_labels": 3},
    "anercorp": {"loader": load_anercorp, "task_type": "ner"},
    "arcd": {"loader": load_arcd, "task_type": "qa"},
    "arcd_50_50": {"loader": load_arcd_50_50, "task_type": "qa"},
    "squad_plus_arcd50": {"loader": load_squad_plus_arcd50, "task_type": "qa"},
    "squad_plus_tydiqa_ar": {"loader": load_squad_plus_tydiqa_ar, "task_type": "qa"},
    "arabic_squad": {"loader": load_arabic_squad, "task_type": "qa"},
    "polyglot_ner_ar": {"loader": load_polyglot_ner_ar, "task_type": "ner"},
}
def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1
def setup_ddp():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def evaluate_classification(model, loader, device, fp16):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attn_mask, labels in loader:
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
            with torch.amp.autocast(device_type="cuda", enabled=fp16):
                logits = model(input_ids, attn_mask)["logits"]
            all_preds += logits.argmax(-1).cpu().tolist()
            all_labels += labels.tolist()
    model.train()
    return classification_metrics(all_preds, all_labels)
def evaluate_ner(model, loader, device, fp16, id2label):
    model.eval()
    pred_tags, gold_tags = [], []
    with torch.no_grad():
        for input_ids, attn_mask, labels in loader:
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
            with torch.amp.autocast(device_type="cuda", enabled=fp16):
                logits = model(input_ids, attn_mask)["logits"]
            preds = logits.argmax(-1).cpu()
            for p_row, l_row in zip(preds, labels):
                p_seq, g_seq = [], []
                for p, l in zip(p_row.tolist(), l_row.tolist()):
                    if l == -100:
                        continue
                    p_seq.append(id2label[p])
                    g_seq.append(id2label[l])
                pred_tags.append(p_seq)
                gold_tags.append(g_seq)
    model.train()
    return seqeval_ner_metrics(pred_tags, gold_tags)
def evaluate_qa(model, loader, device, fp16, tokenizer, max_answer_length=30, n_best=20):
    model.eval()
    preds, golds = {}, {}
    sep_id = tokenizer.sep_token_id
    with torch.no_grad():
        for input_ids, attn_mask, starts, ends, ids, gold in loader:
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
            with torch.amp.autocast(device_type="cuda", enabled=fp16):
                out = model(input_ids, attn_mask)
            s_logits_b = out["start_logits"].float().cpu()
            e_logits_b = out["end_logits"].float().cpu()
            ids_b = input_ids.cpu()
            mask_b = attn_mask.cpu()
            for i, qid in enumerate(ids):
                row = ids_b[i]
                real_len = int(mask_b[i].sum().item())
                sep_pos = (row[:real_len] == sep_id).nonzero(as_tuple=True)[0]
                golds[qid] = gold[i]
                if len(sep_pos) == 0:
                    preds[qid] = []
                    continue
                ctx_start = sep_pos[0].item() + 1
                ctx_end = real_len - 2
                if ctx_end < ctx_start:
                    preds[qid] = []
                    continue
                s_logits, e_logits = s_logits_b[i], e_logits_b[i]
                k = min(n_best, ctx_end - ctx_start + 1)
                start_top = (torch.topk(s_logits[ctx_start:ctx_end+1], k).indices + ctx_start).tolist()
                end_top = (torch.topk(e_logits[ctx_start:ctx_end+1], k).indices + ctx_start).tolist()
                best_score, best_span = float("-inf"), None
                for s in start_top:
                    for e in end_top:
                        if e < s or (e - s + 1) > max_answer_length:
                            continue
                        score = s_logits[s].item() + e_logits[e].item()
                        if score > best_score:
                            best_score, best_span = score, (s, e)
                preds[qid] = row[best_span[0]:best_span[1]+1].tolist() if best_span else []
    model.train()
    return preds, golds
def build_model(task_type, mcfg, num_labels=None):
    if task_type == "classification":
        return MambaForSequenceClassification(mcfg, num_labels=num_labels)
    if task_type == "ner":
        return MambaForTokenClassification(mcfg, num_labels=num_labels)
    if task_type == "qa":
        return MambaForQuestionAnswering(mcfg)
    raise ValueError(f"unknown task_type: {task_type}")
def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    task_name = cfg["task"]["name"]
    task_info = TASK_REGISTRY[task_name]
    task_type = task_info["task_type"]
    ddp = is_distributed()
    if ddp:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
        is_main = local_rank == 0
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True
        world_size = 1
    set_seed(cfg["training"].get("seed", 42))
    tokenizer = AutoTokenizer.from_pretrained(cfg["data"]["tokenizer_name"])
    assert tokenizer.is_fast, "finetune.py requires a fast tokenizer (word_ids/offsets)"
    if is_main:
        pass
    train_ds, dev_ds, test_ds = task_info["loader"](
        tokenizer, cfg["data"]["max_seq_length"]
    )
    if task_type == "ner":
        label2id = train_ds.label2id
        id2label = {v: k for k, v in label2id.items()}
        num_labels = len(label2id)
    else:
        num_labels = task_info.get("num_labels")
    mcfg = dict(cfg["model"])
    mcfg["vocab_size"] = tokenizer.vocab_size
    mcfg["pad_token_id"] = tokenizer.pad_token_id
    model = build_model(task_type, mcfg, num_labels=num_labels).to(device)
    if cfg.get("pretrained_ckpt"):
        load_pretrained_encoder(model.encoder, cfg["pretrained_ckpt"], device=device)
        if is_main:
            pass
    elif is_main:
        print(
            "WARNING: no pretrained_ckpt set in config -- training encoder from "
            "scratch, this defeats the point of the benchmark"
        )
    if ddp:
        model = DDP(model, device_ids=[local_rank])
        core_model = model.module
    else:
        core_model = model
    tcfg = cfg["training"]
    if task_type == "classification":
        collate = lambda b: collate_classification(b, tokenizer.pad_token_id)
    elif task_type == "ner":
        collate = lambda b: collate_ner(b, tokenizer.pad_token_id)
    else:
        collate = lambda b: collate_qa(b, tokenizer.pad_token_id)
    if ddp:
        train_sampler = DistributedSampler(
            train_ds, shuffle=True, seed=tcfg.get("seed", 42)
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=tcfg["batch_size"],
            sampler=train_sampler,
            num_workers=tcfg["num_workers"],
            collate_fn=collate,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_ds,
            batch_size=tcfg["batch_size"],
            shuffle=True,
            num_workers=tcfg["num_workers"],
            collate_fn=collate,
        )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=tcfg["eval_batch_size"],
        shuffle=False,
        num_workers=tcfg["num_workers"],
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=tcfg["eval_batch_size"],
        shuffle=False,
        num_workers=tcfg["num_workers"],
        collate_fn=collate,
    )
    optim = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=tcfg["fp16"])
    total_steps = len(train_loader) * tcfg["epochs"]
    warmup_steps = tcfg.get("warmup_steps", 0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda step: min(1.0, step / max(1, warmup_steps))
        * max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps)),
    )
    if is_main:
        os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
        effective_batch_size = tcfg["batch_size"] * world_size
        print(
            "NOTE: effective batch size scales with world_size (same convention "
            "as train.py) -- if you compare against a single-GPU run of the same "
            "config, the two aren't directly comparable without adjusting lr/epochs."
        )
    best_metric, best_state = -1.0, None
    primary_metric = {"classification": "macro_f1", "ner": "f1", "qa": "f1"}[task_type]
    model.train()
    for epoch in range(tcfg["epochs"]):
        if ddp:
            train_sampler.set_epoch(epoch)
        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{tcfg['epochs']}",
            dynamic_ncols=True,
            disable=not is_main,
        )
        for batch in pbar:
            optim.zero_grad()
            if task_type in ("classification", "ner"):
                input_ids, attn_mask, labels = batch
                input_ids, attn_mask, labels = (
                    input_ids.to(device),
                    attn_mask.to(device),
                    labels.to(device),
                )
                with torch.amp.autocast(device_type="cuda", enabled=tcfg["fp16"]):
                    out = model(input_ids, attn_mask, labels)
            else:
                input_ids, attn_mask, starts, ends, ids, gold = batch
                input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
                starts, ends = starts.to(device), ends.to(device)
                with torch.amp.autocast(device_type="cuda", enabled=tcfg["fp16"]):
                    out = model(input_ids, attn_mask, starts, ends)
            loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            if is_main:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        if ddp:
            dist.barrier()
        if is_main:
            if task_type == "classification":
                metrics = evaluate_classification(
                    model, dev_loader, device, tcfg["fp16"]
                )
            elif task_type == "ner":
                metrics = evaluate_ner(
                    model, dev_loader, device, tcfg["fp16"], id2label
                )
            else:
                preds, golds = evaluate_qa(model, dev_loader, device, tcfg["fp16"], tokenizer)
                preds_text = {
                    qid: tokenizer.decode(ids, skip_special_tokens=True)
                    for qid, ids in preds.items()
                }
                metrics = qa_metrics(preds_text, golds)
            if metrics[primary_metric] > best_metric:
                best_metric = metrics[primary_metric]
                best_state = {
                    k: v.cpu().clone() for k, v in core_model.state_dict().items()
                }
                print(
                    f"  -> new best ({primary_metric}={best_metric:.4f}), checkpointing in memory"
                )
        if ddp:
            dist.barrier()
    if is_main:
        if best_state is not None:
            core_model.load_state_dict(best_state)
        if task_type == "classification":
            test_metrics = evaluate_classification(
                model, test_loader, device, tcfg["fp16"]
            )
        elif task_type == "ner":
            test_metrics = evaluate_ner(
                model, test_loader, device, tcfg["fp16"], id2label
            )
        else:
            preds, golds = evaluate_qa(model, test_loader, device, tcfg["fp16"], tokenizer)
            preds_text = {
                qid: tokenizer.decode(ids, skip_special_tokens=True)
                for qid, ids in preds.items()
            }
            test_metrics = qa_metrics(preds_text, golds)
        print(
            f"=== FINAL test metrics for {task_name} "
            f"(seed={tcfg.get('seed', 42)}): {test_metrics} ==="
        )
        result_path = os.path.join(
            cfg["paths"]["output_dir"], f"{task_name}_seed{tcfg.get('seed', 42)}.json"
        )
        with open(result_path, "w") as f:
            json.dump(
                {
                    "task": task_name,
                    "seed": tcfg.get("seed", 42),
                    "world_size": world_size,
                    "dev_best": best_metric,
                    "test": test_metrics,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        if best_state is not None:
            ckpt_path = os.path.join(cfg["paths"]["output_dir"], f"{task_name}_best.pt")
            torch.save(best_state, ckpt_path)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)
```


## src/models/mamba-old.py

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from mamba_ssm import Mamba
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    MAMBA_SSM_AVAILABLE = False
class NaiveSSM(nn.Module):
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
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
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
        self.decoder.weight = self.encoder.word_emb.weight
        self.apply(self._init_weights)
    @staticmethod
    def _init_weights(module):
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


## src/models/mamba.py

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from mamba_ssm import Mamba
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    MAMBA_SSM_AVAILABLE = False
def _flip_valid(x, attention_mask):
    B, L, D = x.shape
    lengths = attention_mask.sum(dim=1).long()
    idx = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L).clone()
    for b in range(B):
        n = lengths[b].item()
        if n > 0:
            idx[b, :n] = torch.arange(n - 1, -1, -1, device=x.device)
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, D))
class NaiveSSM(nn.Module):
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
        if attention_mask is not None:
            bwd_in = _flip_valid(h, attention_mask)
            bwd = _flip_valid(self.bwd_mixer(bwd_in), attention_mask)
        else:
            bwd_in = h.flip(dims=[1])
            bwd = self.bwd_mixer(bwd_in).flip(dims=[1])
        merged = self.merge(torch.cat([fwd, bwd], dim=-1))
        x = residual + self.dropout(merged)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
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
        self.decoder.weight = self.encoder.word_emb.weight
        self.apply(self._init_weights)
    @staticmethod
    def _init_weights(module):
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


## src/models/mamba_heads.py

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba import MambaEncoder, MambaForMaskedLM
def load_pretrained_encoder(encoder, ckpt_path, device="cpu", strict=False):
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
        pass
    if unexpected:
        pass
    return encoder
def _mean_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts
class MambaForSequenceClassification(nn.Module):
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
```


## src/sanity/test.py

```py
import torch
from models.mamba import MambaForMaskedLM
torch.manual_seed(0)
device = "cuda"
config = {
    "vocab_size": 1000,
    "d_model": 64,
    "n_layer": 2,
    "d_state": 16,
    "d_conv": 4,
    "expand": 2,
    "pad_token_id": 0,
    "dropout": 0.0,
}
model = MambaForMaskedLM(config).to(device).eval()
B, L = 2, 20
real_len = 10
input_ids_a = torch.randint(1, 1000, (B, L), device=device)
attn_mask = torch.zeros(B, L, device=device)
attn_mask[:, :real_len] = 1
input_ids_a[:, real_len:] = 0
input_ids_b = input_ids_a.clone()
input_ids_b[:, real_len:] = torch.randint(1, 1000, (B, L - real_len), device=device)
with torch.no_grad():
    out_a = model(input_ids_a, attn_mask)["logits"]
    out_b = model(input_ids_b, attn_mask)["logits"]
real_diff = (out_a[:, :real_len] - out_b[:, :real_len]).abs().max().item()
```


## src/train.py

```py
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
        ckpt = torch.load(latest_path, map_location=device)
        model.module.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        scaler.load_state_dict(ckpt["scaler"])
        scheduler.load_state_dict(ckpt["scheduler"])
        opt_step = ckpt["step"]
        if is_main:
            pass
    world_size = dist.get_world_size()
    effective_batch_size = tcfg["batch_size"] * world_size * tcfg["grad_accum_steps"]
    if is_main:
        pass
    start_time = time.time()
    step_start_time = time.time()
    data_start_time = time.time()
    avg_loss = 0.0
    avg_data_time = 0.0
    avg_step_time = 0.0
    model.train()
    while opt_step < total_steps:
        sampler.set_epoch(opt_step)
        pbar = tqdm(
            loader,
            desc=f"⚙️ Step {opt_step}/{total_steps}",
            disable=not is_main,
            dynamic_ncols=True,
        )
        for input_ids, attn_mask, labels in pbar:
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


## src/utils/aggregate_results.py

```py
import argparse
import glob
import json
import os
import statistics
from collections import defaultdict
def load_results(results_dir):
    by_task = defaultdict(list)
    for path in glob.glob(os.path.join(results_dir, "*", "*_seed*.json")):
        with open(path) as f:
            r = json.load(f)
        by_task[r["task"]].append(r)
    return by_task
def summarize(results):
    metrics = defaultdict(list)
    for r in results:
        for k, v in r["test"].items():
            metrics[k].append(v)
    summary = {}
    for k, vals in metrics.items():
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        summary[k] = (mean, std, len(vals))
    return summary
def main(results_dir):
    by_task = load_results(results_dir)
    if not by_task:
        return
    for task, results in sorted(by_task.items()):
        summary = summarize(results)
        for metric, (mean, std, n) in sorted(summary.items()):
            flag = "  (single run, not a variance estimate)" if n == 1 else ""
    n_seeds = {task: len(r) for task, r in by_task.items()}
    under_powered = [t for t, n in n_seeds.items() if n < 3]
    if under_powered:
        print(
            f"\nNote: {under_powered} have fewer than 3 seeds -- run "
            f"`make benchmark-seeds TASK=<name>` before reporting these as final."
        )
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results_finetune")
    args = p.parse_args()
    main(args.results_dir)
```


## src/utils/eval.py

```py
import math
import torch
@torch.no_grad()
def evaluate(model, val_loader, device, fp16, max_batches=None):
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
    perplexity = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    return {"loss": mean_loss, "accuracy": accuracy, "perplexity": perplexity}
```


## src/utils/finetune_metrics.py

```py
import collections
import re
import string
import numpy as np
def classification_metrics(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    acc = (preds == labels).mean()
    classes = sorted(set(labels.tolist()) | set(preds.tolist()))
    f1s = []
    for c in classes:
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return {"accuracy": float(acc), "macro_f1": float(np.mean(f1s))}
def seqeval_ner_metrics(pred_tags, gold_tags):
    try:
        from seqeval.metrics import f1_score, precision_score, recall_score
        return {
            "precision": precision_score(gold_tags, pred_tags),
            "recall": recall_score(gold_tags, pred_tags),
            "f1": f1_score(gold_tags, pred_tags),
        }
    except ImportError:
        correct = sum(
            p == g
            for pseq, gseq in zip(pred_tags, gold_tags)
            for p, g in zip(pseq, gseq)
        )
        total = sum(len(gseq) for gseq in gold_tags)
        print(
            "[finetune_metrics] seqeval not installed -- reporting TOKEN accuracy, "
            "which is NOT comparable to published entity-level F1. "
            "Run: pip install seqeval --break-system-packages"
        )
        return {"token_accuracy": correct / max(1, total)}
_AR_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")
def _normalize_arabic_answer(s):
    s = _AR_DIACRITICS.sub("", s)
    s = "".join(ch for ch in s if ch not in string.punctuation and ch not in "،؛؟”“")
    s = re.sub(r"\s+", " ", s).strip()
    return s
def qa_metrics(preds, golds):
    em_total, f1_total = 0.0, 0.0
    for qid, gold_list in golds.items():
        pred = _normalize_arabic_answer(preds.get(qid, ""))
        best_em, best_f1 = 0.0, 0.0
        for gold in gold_list:
            gold_n = _normalize_arabic_answer(gold)
            em = float(pred == gold_n)
            pred_toks, gold_toks = pred.split(), gold_n.split()
            common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
            num_same = sum(common.values())
            if num_same == 0:
                f1 = 0.0
            else:
                prec = num_same / max(1, len(pred_toks))
                rec = num_same / max(1, len(gold_toks))
                f1 = 2 * prec * rec / (prec + rec)
            best_em, best_f1 = max(best_em, em), max(best_f1, f1)
        em_total += best_em
        f1_total += best_f1
    n = max(1, len(golds))
    return {"exact_match": 100 * em_total / n, "f1": 100 * f1_total / n}
```


## test.py

```py
from datasets import load_dataset
ds = load_dataset("asas-ai/ANERCorp")
```

