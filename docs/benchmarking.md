# Benchmark testing guide

Evaluates the pretrained Bi-Mamba encoder (`checkpoint/latest.pt`, 200k steps)
on 4 MSA-only downstream tasks, matching the evaluation set AraBERT itself
reports on (sentiment / NER / QA), plus one NLI task. Dialect evaluation is
explicitly out of scope here — see `paths.output_dir` naming, it's kept
separate from anything dialect-related on purpose.

## 1. New files (drop into your existing tree)

```
src/models/mamba_heads.py        # classification / token-classification / QA heads
src/data/finetune_datasets.py    # HARD, XNLI-ar, ANERcorp, ARCD loaders
src/utils/finetune_metrics.py    # accuracy/F1, seqeval NER F1, SQuAD-style EM/F1
src/finetune.py                  # single-GPU finetune + eval driver
src/utils/aggregate_results.py   # multi-seed mean+-std table generator
configs/finetune_hard.yaml
configs/finetune_xnli.yaml
configs/finetune_anercorp.yaml
configs/finetune_arcd.yaml
Makefile                         # benchmark-* targets, see section 3 below
```

The `Makefile` below appends a `Benchmark phase` section to your existing
one (`prepare-data`/`train`/`dry-test`/`codebase` targets untouched), using
the same `mkdir -p $(LOG_DIR)` + `tee` logging convention you already use
for `prepare-data`/`train`.

They plug into your existing `src/models/mamba.py` and `src/utils/eval.py`
without modifying either — `mamba_heads.py` only imports `MambaEncoder`
and `MambaForMaskedLM._init_weights` from `mamba.py`.

## 2. One-time setup

```bash
make benchmark-deps              # pip install datasets seqeval
make benchmark-check-tokenizer   # confirms AraBERT tokenizer is a *Fast* tokenizer
```

`datasets` streaming isn't needed here (these are small, non-streamed
downloads unlike `prepare_data.py`'s corpora) — plain `load_dataset(...)`
is used throughout `finetune_datasets.py`.

**Before your first real run**, also verify the ANERcorp column names —
this was written without live access to the HF dataset viewer, so double
check:

```python
from datasets import load_dataset
ds = load_dataset("asas-ai/ANERCorp")
print(ds["train"].features)
```

and fix the `TOKENS_COL` / `TAGS_COL` constants in `load_anercorp()` if they
don't match `"tokens"` / `"ner_tags"`.

## 3. Running the suite

Each task is a separate, independent single-GPU job — no DDP, these
datasets are small enough that multi-GPU would only add complexity.

Single task:
```bash
make benchmark-hard
make benchmark-xnli
make benchmark-anercorp
make benchmark-arcd
```

All 4 in parallel, one per GPU (matches your 4x2080Ti layout):
```bash
make benchmark-all
```

Every target follows the same `mkdir -p logs/` + `tee logs/benchmark_<task>_<timestamp>.log`
convention as `make train`, so a run's full stdout is always saved, not
just what scrolled past in the terminal.

Each run writes `results_finetune/<task>/<task>_seed<seed>.json` with dev
and test metrics, plus a `<task>_best.pt` checkpoint (best-dev-epoch state
dict of the *task head*, not the pretraining checkpoint).

## 4. Multiple seeds (do this before publishing anything)

Fine-tuning variance on small MSA datasets (ANERcorp ~5k sentences, ARCD
~700 train examples) is large enough that a single run is not a
trustworthy number. Run each task 3x with different seeds and report
mean ± std, not a cherry-picked best run:

```bash
make benchmark-seeds TASK=hard        # runs seeds 42, 1337, 2024 by default
make benchmark-seeds TASK=xnli
make benchmark-seeds TASK=anercorp
make benchmark-seeds TASK=arcd

# or all four, sequentially:
make benchmark-seeds-all

# override the seed list if you want a different set:
make benchmark-seeds TASK=hard SEEDS="1 2 3 4 5"
```

Once you have result files, aggregate them into a mean±std markdown table:

```bash
make benchmark-aggregate
```

This runs `src/utils/aggregate_results.py`, which scans
`results_finetune/*/*_seed*.json` and prints a table (also flags any task
with fewer than 3 seeds so you don't accidentally report an underpowered
result as final).

To start over on the benchmark phase without touching your pretraining
checkpoint or logs:
```bash
make benchmark-clean
```

## 5. What to report in the paper

| Task | Dataset | Metric | Your model | AraBERT (published) | Notes |
|---|---|---|---|---|---|
| Sentiment | HARD | Acc / macro-F1 | — | AraBERTv02 baseline | reviews are MSA-leaning, not pure MSA — say so |
| NLI | XNLI-ar | Accuracy | — | — (XNLI-ar not in original AraBERT eval) | translated, not native MSA |
| NER | ANERcorp | Entity-level F1 (seqeval) | — | ~similar setup in AraBERT paper | direct comparison point |
| QA | ARCD | EM / F1 | — | AraBERTv02 baseline | same Wikipedia domain as your pretraining data |

Fill "AraBERT (published)" from the AraBERT paper/model card rather than
re-running AraBERT yourself unless you have spare compute — but state
explicitly that it's a **published number, not a compute-matched
re-run**, since your model (512d/12L, ~120GB corpus, 4x2080Ti) and AraBERT
(768d/12L, 77GB corpus, TPU-scale compute) aren't parameter- or
compute-matched. That asymmetry belongs in the paper as a sentence, not
buried in a footnote — same instinct you already show flagging the
fp16-not-bf16 Turing constraint in `configs/base.yaml`.

Also report, per task: wall-clock finetune time and peak VRAM (you
already have the pattern for this in `train.py`'s `torch.cuda.max_memory_allocated`
— same call works inside `finetune.py`'s eval loop if you want to add it).
Given the whole framing of this project is "what's achievable on a
4x2080Ti workstation," that's a real result independent of the raw scores.

## 6. Explicit scope note (put this in the paper intro, not just here)

This benchmark suite is MSA-only by construction — HARD (hotel reviews,
MSA-leaning), XNLI-ar (translated MSA), ANERcorp (MSA newswire), ARCD
(MSA Wikipedia). No dialect identification, dialectal sentiment, or
code-switched text is evaluated here. Don't let a reader infer dialectal
competence from these numbers — that's reserved for the follow-up paper.
