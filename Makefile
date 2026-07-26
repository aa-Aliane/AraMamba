PYTHON      ?= python
CONFIG      ?= configs/base.yaml
NPROC       ?= 4
TIMESTAMP   := $(shell date +%Y%m%d_%H%M%S)
LOG_DIR     := logs

.PHONY: all prepare-data train clean-logs dry-test-data dry-test-train dry-test codebase \
        benchmark-deps benchmark-check-tokenizer \
        benchmark-hard benchmark-xnli benchmark-anercorp benchmark-arcd \
        benchmark-all benchmark-seeds benchmark-seeds-all \
        benchmark-aggregate benchmark-clean

all: prepare-data train

TARGET_SIZE_GB ?= 120
WORKERS        ?= 64

prepare-data:
	mkdir -p $(LOG_DIR)
	$(PYTHON) src/data/prepare_data.py --target_size_gb $(TARGET_SIZE_GB) --workers $(WORKERS) 2>&1 | tee $(LOG_DIR)/prepare_data_$(TIMESTAMP).log

train3:
	mkdir -p $(LOG_DIR)
	CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 src/train.py --config $(CONFIG) 2>&1 | tee $(LOG_DIR)/train_3gpus_$(TIMESTAMP).log

train:
	mkdir -p $(LOG_DIR)
	torchrun --nproc_per_node=$(NPROC) src/train.py --config $(CONFIG) 2>&1 | tee $(LOG_DIR)/train_$(TIMESTAMP).log

dry-test-data:
	mkdir -p $(LOG_DIR)
	$(PYTHON) src/data/prepare_data.py --target_size_gb 0.05 --workers 8 2>&1 | tee $(LOG_DIR)/dry_prepare_$(TIMESTAMP).log

dry-test-train:
	mkdir -p $(LOG_DIR)
	torchrun --nproc_per_node=1 src/train.py --config configs/dry_test.yaml 2>&1 | tee $(LOG_DIR)/dry_train_$(TIMESTAMP).log

dry-test: dry-test-data dry-test-train

clean-logs:
	rm -f $(LOG_DIR)/*.log

codebase:
	CodeWeaver -input  . -include ".py,.yaml" -ignore "\.zed,\.ropeproject,.*\txt,.pkl,.*\.git,\.gitignore,.*\.md"

# ---------------------------------------------------------------------------
# Benchmark phase (see docs/BENCHMARKING.md)
# Finetunes + evaluates the pretrained checkpoint on the MSA benchmark
# suite: HARD (sentiment), XNLI-ar (NLI), ANERcorp (NER), ARCD (QA).
# Each task is single-GPU (these datasets are small -- no DDP needed).
# ---------------------------------------------------------------------------
BENCHMARK_RESULTS_DIR ?= results_finetune
SEEDS                 ?= 42 1337 2024
CKPT                   ?= checkpoint/latest.pt

## One-time setup: extra deps finetune.py needs beyond train.py's
benchmark-deps:
	pip install datasets seqeval --break-system-packages

## Fast tokenizer is required for NER word_ids()/QA offset_mapping -- check once
benchmark-check-tokenizer:
	$(PYTHON) -c "from transformers import AutoTokenizer; \
t = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabertv02'); \
assert t.is_fast, 'tokenizer is not fast -- finetune.py will fail on NER/QA'; \
print('tokenizer OK: is_fast =', t.is_fast)"

benchmark-hard:
	mkdir -p $(LOG_DIR)
	$(PYTHON) src/finetune.py --config configs/finetune_hard.yaml 2>&1 | tee $(LOG_DIR)/benchmark_hard_$(TIMESTAMP).log

benchmark-xnli:
	mkdir -p $(LOG_DIR)
	$(PYTHON) src/finetune.py --config configs/finetune_xnli.yaml 2>&1 | tee $(LOG_DIR)/benchmark_xnli_$(TIMESTAMP).log

benchmark-anercorp:
	mkdir -p $(LOG_DIR)
	$(PYTHON) src/finetune.py --config configs/finetune_anercorp.yaml 2>&1 | tee $(LOG_DIR)/benchmark_anercorp_$(TIMESTAMP).log

benchmark-arcd:
	mkdir -p $(LOG_DIR)
	$(PYTHON) src/finetune.py --config configs/finetune_arcd.yaml 2>&1 | tee $(LOG_DIR)/benchmark_arcd_$(TIMESTAMP).log

## All 4 tasks in parallel, one per GPU -- matches the 4x2080Ti layout.
## Each job also writes its own per-task tee log to $(LOG_DIR); this
## target's log captures the interleaved stdout of the `wait`.
benchmark-all:
	mkdir -p $(LOG_DIR)
	( CUDA_VISIBLE_DEVICES=0 $(PYTHON) src/finetune.py --config configs/finetune_hard.yaml 2>&1 | tee $(LOG_DIR)/benchmark_hard_$(TIMESTAMP).log & \
	  CUDA_VISIBLE_DEVICES=1 $(PYTHON) src/finetune.py --config configs/finetune_xnli.yaml 2>&1 | tee $(LOG_DIR)/benchmark_xnli_$(TIMESTAMP).log & \
	  CUDA_VISIBLE_DEVICES=2 $(PYTHON) src/finetune.py --config configs/finetune_anercorp.yaml 2>&1 | tee $(LOG_DIR)/benchmark_anercorp_$(TIMESTAMP).log & \
	  CUDA_VISIBLE_DEVICES=3 $(PYTHON) src/finetune.py --config configs/finetune_arcd.yaml 2>&1 | tee $(LOG_DIR)/benchmark_arcd_$(TIMESTAMP).log & \
	  wait ) 2>&1 | tee $(LOG_DIR)/benchmark_all_$(TIMESTAMP).log

## Multi-seed sweep for ONE task: make benchmark-seeds TASK=hard
## TASK must match a configs/finetune_<TASK>.yaml filename.
benchmark-seeds:
	@if [ -z "$(TASK)" ]; then \
		echo "usage: make benchmark-seeds TASK=hard|xnli|anercorp|arcd"; exit 1; \
	fi
	mkdir -p $(LOG_DIR)
	@for seed in $(SEEDS); do \
		echo "=== $(TASK) seed=$$seed ==="; \
		$(PYTHON) -c "import yaml; \
cfg = yaml.safe_load(open('configs/finetune_$(TASK).yaml')); \
cfg['training']['seed'] = $$seed; \
yaml.dump(cfg, open('configs/finetune_$(TASK)_seed$$seed.yaml', 'w'))"; \
		$(PYTHON) src/finetune.py --config configs/finetune_$(TASK)_seed$$seed.yaml 2>&1 | tee $(LOG_DIR)/benchmark_$(TASK)_seed$$seed_$(TIMESTAMP).log; \
	done

## Multi-seed sweep for all 4 tasks, sequentially
benchmark-seeds-all:
	$(MAKE) benchmark-seeds TASK=hard
	$(MAKE) benchmark-seeds TASK=xnli
	$(MAKE) benchmark-seeds TASK=anercorp
	$(MAKE) benchmark-seeds TASK=arcd

## Aggregate all result JSONs into a mean+-std markdown table
benchmark-aggregate:
	$(PYTHON) src/utils/aggregate_results.py --results_dir $(BENCHMARK_RESULTS_DIR)

## Wipe benchmark outputs and generated per-seed configs
## (does NOT touch checkpoint/, outputs/, or logs/ from pretraining)
benchmark-clean:
	rm -rf $(BENCHMARK_RESULTS_DIR)
	rm -f configs/finetune_*_seed*.yaml
