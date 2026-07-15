PYTHON      ?= python
CONFIG      ?= configs/base.yaml
NPROC       ?= 4
TIMESTAMP   := $(shell date +%Y%m%d_%H%M%S)
LOG_DIR     := logs

.PHONY: all prepare-data train clean-logs dry-test-data dry-test-train dry-test

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
