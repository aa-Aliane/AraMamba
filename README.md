---
language:
- ar
license: apache-2.0
tags:
- mamba
- state-space-model
- arabic
- masked-language-modeling
- bidirectional
datasets:
- wikimedia/wikipedia
- uonlp/CulturaX
pipeline_tag: fill-mask
---

# AraSSM-base

AraSSM is a bidirectional state-space (Mamba) encoder pretrained from scratch for Arabic via
masked language modeling. It is, to our knowledge, the first bidirectional Mamba/SSM encoder
pretrained specifically for Arabic, and was trained entirely on four consumer-grade NVIDIA RTX
2080Ti GPUs (11GB) rather than an accelerator cluster.

## Model description

Each AraSSM layer runs a forward and a backward selective-scan (Mamba) mixer over the same
input and merges the two outputs, giving the model full bidirectional context while keeping
$O(L)$ complexity in sequence length $L$, instead of the $O(L^2)$ complexity of self-attention.

| | |
|---|---|
| Layers | 12 |
| Hidden size ($d_{\text{model}}$) | 512 |
| State dimension | 16 |
| Parameters | ~105M |
| Max sequence length | 512 |
| Vocabulary size | 64,000 |

## Tokenizer

AraSSM uses the existing **AraBERTv02 tokenizer**
([`aubmindlab/bert-base-arabertv02`](https://huggingface.co/aubmindlab/bert-base-arabertv02))
rather than a custom-trained one. This tokenizer was not trained as part of this work; it is
reused as-is, both to decouple corpus preprocessing from tokenizer choice and to keep results
comparable to AraBERT-family baselines that use the same vocabulary. All credit for the
tokenizer belongs to its original authors.

## Training data

AraSSM is pretrained on a cleaned, deduplicated corpus of approximately 80GB of Arabic text
(79.6GB train / 0.4GB validation), combining Arabic Wikipedia and the Arabic portion of
CulturaX. Documents are stripped of diacritics and URLs, filtered for length and Arabic-script
ratio, chunked to at most 400 words, and deduplicated at the chunk level with a Bloom filter.

## Training procedure

- Objective: standard BERT-style masked language modeling (15% masking, 80/10/10 split)
- Optimizer: AdamW, lr 3e-4, weight decay 0.01, 10,000 warmup steps, linear decay
- Precision: fp16 (Turing GPUs do not support accelerated bf16)
- Effective batch size: 256 (batch size 8 x grad accumulation 8 x 4 GPUs)
- Compute: 4x RTX 2080Ti, ~960 GPU-hours (~10 days)

## How to use

AraSSM is a custom architecture (not a native `transformers` model class), so loading it
requires the model code from the project repository:

```python
from huggingface_hub import PyTorchModelHubMixin
from models.mamba import MambaForMaskedLM  # from the AraSSM repository

class HubMambaForMaskedLM(MambaForMaskedLM, PyTorchModelHubMixin):
    pass

model = HubMambaForMaskedLM.from_pretrained("aliane29/arassm-base")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("aliane29/arassm-base")
```

## Intended use

This is a pretrained encoder intended to be fine-tuned on downstream Arabic NLU tasks
(classification, token classification, extractive question answering), similarly to how a
BERT-family encoder is used. It has not been fine-tuned for any specific task in this repository.

## Limitations

- Pretrained on Modern Standard Arabic and web text (Wikipedia + CulturaX); performance on
  dialectal Arabic is not evaluated.
- Maximum sequence length is 512 tokens.
- Trained on a fixed, publicly available compute budget (4 consumer GPUs); larger-scale
  Transformer baselines were pretrained on substantially larger accelerator-cluster budgets.

## Citation

If you use this model, please cite:

```bibtex
@misc{arassm2026,
  title  = {AraSSM: A Bidirectional State-Space Encoder for Arabic Masked Language Modeling},
  author = {Aliane, Ahmed Amine and Aliane, Hassina and Semmar, Nasredine},
  year   = {2026},
  note   = {Preprint}
}
```
