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
    dist.init_process_group(backend="nccl")
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

                if is_main and opt_step % tcfg["save_steps"] == 0 and opt_step > 0:
                    ckpt = {
                        "step": opt_step,
                        "model": model.module.state_dict(),
                        "optim": optim.state_dict(),
                        "scaler": scaler.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }
                    ckpt_path = f"{cfg['paths']['checkpoint_dir']}/step_{opt_step}.pt"
                    torch.save(ckpt, ckpt_path)
                    # also a fixed-name "latest" pointer so resume doesn't need to scan filenames
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
