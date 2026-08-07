PYTHON      ?= python
CONFIG      ?= configs/base.yaml
NPROC       ?= 4
TIMESTAMP   := $(shell date +%Y%m%d_%H%M%S)
LOG_DIR     := logs

TARGET_SIZE_GB ?= 120
WORKERS        ?= 64

ABLATION_RESULTS_DIR ?= results_finetune/ablation_nopretrain
ABLATION_CONFIG_PREFIX ?= ablation_nopretrain

HUB_REPO_ID ?=aliane29/arassm-base
HUB_CONFIG  ?= configs/base.yaml
HUB_CKPT    ?= checkpoint/latest.pt


.PHONY: all prepare-data train clean-logs dry-test-data dry-test-train dry-test codebase \
        benchmark-deps benchmark-check-tokenizer \
        benchmark-hard benchmark-xnli benchmark-anercorp benchmark-arcd \
        benchmark-all benchmark-seeds benchmark-seeds-all \
        benchmark-aggregate benchmark-clean \
        prefinetune-arcd prefinetune-anercorp \
        ablation-seeds ablation-seeds-all ablation-compare ablation-clean \
        push-to-hub

all: prepare-data train



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
	CodeWeaver -input  . -include ".py,.yaml" -ignore "\.zed,\.ropeproject,.txt,.pkl,\.git,\.gitignore,__init__\.py,__pycache__" -output codebase.md

## Pretrain ARCD's encoder on Arabic Wikipedia (1.5M sentences).
pretrain-arcd:
	mkdir -p $(LOG_DIR)
	$(PYTHON) src/pretrain.py --config configs/pretrain_arcd.yaml 2>&1 | tee $(LOG_DIR)/pretrain_arcd_$(TIMESTAMP).log

## Pretrain ANERcorp's encoder on Arabic Wikipedia (1.5M sentences).
pretrain-anercorp:
	mkdir -p $(LOG_DIR)
	$(PYTHON) src/pretrain.py --config configs/pretrain_anercorp.yaml "


## Prefinetune ARCD's encoder on Arabic-SQuAD (48k QA pairs) before the
## tiny ARCD gold set. Run this once; finetune_arcd.yaml already points
## pretrained_ckpt at its output (results_finetune/prefinetune_arcd/arabic_squad_best.pt).
prefinetune-arcd:
	mkdir -p $(LOG_DIR)
	$(PYTHON) src/finetune.py --config configs/prefinetune_arcd.yaml 2>&1 | tee $(LOG_DIR)/prefinetune_arcd_$(TIMESTAMP).log

prefinetune-arcd-v2:
	mkdir -p $(LOG_DIR)
	$(PYTHON) src/finetune.py --config configs/prefinetune_arcd_v2.yaml 2>&1 | tee $(LOG_DIR)/prefinetune_arcd_v2_$(TIMESTAMP).log

## Prefinetune ANERcorp's encoder on Polyglot-NER Arabic (silver Wikipedia
## NER, capped at 150k sentences) before the small ANERcorp gold set.
## finetune_anercorp.yaml already points pretrained_ckpt at its output
## (results_finetune/prefinetune_anercorp/polyglot_ner_ar_best.pt).
prefinetune-anercorp:
	mkdir -p $(LOG_DIR)
	$(PYTHON) src/finetune.py --config configs/prefinetune_anercorp.yaml 2>&1 | tee $(LOG_DIR)/prefinetune_anercorp_$(TIMESTAMP).log

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
	torchrun --nproc_per_node=$(NPROC) src/finetune.py --config configs/finetune_hard.yaml 2>&1 | tee $(LOG_DIR)/benchmark_hard_$(TIMESTAMP).log

benchmark-xnli:
	mkdir -p $(LOG_DIR)
	torchrun --nproc_per_node=$(NPROC) src/finetune.py --config configs/finetune_xnli.yaml 2>&1 | tee $(LOG_DIR)/benchmark_xnli_$(TIMESTAMP).log

benchmark-anercorp:
	mkdir -p $(LOG_DIR)
	$(PYTHON) src/finetune.py --config configs/finetune_anercorp.yaml 2>&1 | tee $(LOG_DIR)/benchmark_anercorp_$(TIMESTAMP).log

benchmark-arcd:
	mkdir -p $(LOG_DIR)
	$(PYTHON) src/finetune.py --config configs/finetune_arcd.yaml 2>&1 | tee $(LOG_DIR)/benchmark_arcd_$(TIMESTAMP).log

benchmark-squad-plus-arcd50:
	mkdir -p $(LOG_DIR)
	$(PYTHON) src/finetune.py --config configs/finetune_squad_plus_arcd50.yaml 2>&1 | tee $(LOG_DIR)/benchmark_squad_plus_arcd50_$(TIMESTAMP).log

# benchmark-arcd:
# 	mkdir -p $(LOG_DIR)
# 	torchrun --nproc_per_node=$(NPROC) src/finetune.py --config configs/finetune_arcd.yaml 2>&1 | tee $(LOG_DIR)/benchmark_arcd_$(TIMESTAMP).log

## All 4 tasks, one after another.
## HARD/XNLI use all $(NPROC) GPUs via DDP (large enough datasets that
## DDP's ÷N step-count hit doesn't matter and the wall-clock win is real).
## ANERcorp/ARCD run single-GPU (DDP was confirmed to starve them of
## gradient steps -- e.g. ARCD dropped to 22 steps/epoch under 4-way DDP
## vs ~87 on a single GPU -- and both finish in a couple minutes anyway).
benchmark-all: benchmark-hard benchmark-xnli benchmark-anercorp benchmark-arcd

## Task -> launcher mapping used by benchmark-seeds below.
DDP_TASKS := hard xnli

## Multi-seed sweep for ONE task: make benchmark-seeds TASK=hard
## TASK must match a configs/finetune_<TASK>.yaml filename. Automatically
## uses torchrun for hard/xnli and plain python for anercorp/arcd, same
## split as the individual targets above.
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
		if echo "$(DDP_TASKS)" | grep -qw "$(TASK)"; then \
			torchrun --nproc_per_node=$(NPROC) src/finetune.py --config configs/finetune_$(TASK)_seed$$seed.yaml 2>&1 | tee $(LOG_DIR)/benchmark_$(TASK)_seed$$seed_$(TIMESTAMP).log; \
		else \
			$(PYTHON) src/finetune.py --config configs/finetune_$(TASK)_seed$$seed.yaml 2>&1 | tee $(LOG_DIR)/benchmark_$(TASK)_seed$$seed_$(TIMESTAMP).log; \
		fi; \
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



ablation-seeds:
	@if [ -z "$(TASK)" ]; then \
		echo "usage: make ablation-seeds TASK=hard|xnli|anercorp|arcd"; exit 1; \
	fi
	mkdir -p $(LOG_DIR)
	@for seed in $(SEEDS); do \
		echo "=== ablation no-pretrain $(TASK) seed=$$seed ==="; \
		$(PYTHON) -c "import yaml; \
cfg = yaml.safe_load(open('configs/$(ABLATION_CONFIG_PREFIX)_$(TASK).yaml')); \
cfg['training']['seed'] = $$seed; \
yaml.dump(cfg, open('configs/$(ABLATION_CONFIG_PREFIX)_$(TASK)_seed$$seed.yaml', 'w'))" || exit 1; \
		if echo "$(DDP_TASKS)" | grep -qw "$(TASK)"; then \
			torchrun --nproc_per_node=$(NPROC) src/finetune.py --config configs/$(ABLATION_CONFIG_PREFIX)_$(TASK)_seed$$seed.yaml 2>&1 | tee $(LOG_DIR)/ablation_$(TASK)_seed$$seed_$(TIMESTAMP).log || exit 1; \
		else \
			$(PYTHON) src/finetune.py --config configs/$(ABLATION_CONFIG_PREFIX)_$(TASK)_seed$$seed.yaml 2>&1 | tee $(LOG_DIR)/ablation_$(TASK)_seed$$seed_$(TIMESTAMP).log || exit 1; \
		fi; \
	done

## Multi-seed no-pretrain ablation sweep for all 4 tasks, sequentially
ablation-seeds-all:
	$(MAKE) ablation-seeds TASK=hard
	$(MAKE) ablation-seeds TASK=xnli
	$(MAKE) ablation-seeds TASK=anercorp
	$(MAKE) ablation-seeds TASK=arcd

## Markdown table comparing pretrained vs no-pretrain results, per task/metric
ablation-compare:
	$(PYTHON) src/utils/compare_ablation.py --pretrained_dir $(BENCHMARK_RESULTS_DIR) --ablation_dir $(ABLATION_RESULTS_DIR)

## Wipe ablation outputs and generated per-seed configs
## (does NOT touch checkpoint/, outputs/, logs/, or results_finetune/<task> from the main benchmark)
ablation-clean:
	rm -rf $(ABLATION_RESULTS_DIR)
	rm -f configs/$(ABLATION_CONFIG_PREFIX)_*_seed*.yaml





## Push the pretrained AraSSM encoder (+ tokenizer) to the Hugging Face Hub.
## Private by default -- pass HUB_PUBLIC=1 to push publicly instead.
## Usage: make push-to-hub HUB_REPO_ID=your-username/arassm-base
push-to-hub:
	@if [ -z "$(HUB_REPO_ID)" ]; then \
		echo "usage: make push-to-hub HUB_REPO_ID=your-username/arassm-base"; exit 1; \
	fi
	cd src && $(PYTHON) push_to_hub.py \
		--config ../$(HUB_CONFIG) \
		--checkpoint ../$(HUB_CKPT) \
		--repo_id $(HUB_REPO_ID) \
		--push_tokenizer \
		$(if $(HUB_PUBLIC),--public,)
