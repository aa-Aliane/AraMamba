"""
Multi-GPU (DDP) finetuning + evaluation driver for the MSA benchmark
suite (see docs/BENCHMARKING.md). Mirrors train.py's DDP setup so a
single task -- HARD's ~105k rows in particular -- can use all 4 GPUs
instead of just one.

Launch two ways:
  single GPU, no DDP:
    python src/finetune.py --config configs/finetune_hard.yaml
  multi GPU, DDP (what make benchmark-* now uses):
    torchrun --nproc_per_node=4 src/finetune.py --config configs/finetune_hard.yaml

Only rank 0 evaluates on dev/test and writes results/checkpoints -- the
same "only rank 0 evaluates" pattern train.py already uses for its
val_loader, since eval here is a no_grad forward pass with no
backward/allreduce needed. This works safely with a DDP-wrapped model
specifically because this encoder has no BatchNorm-style buffers to
sync (LayerNorm only) -- if you ever add a buffer-holding layer, revisit
this assumption.

NOTE: with DDP, every rank independently calls load_dataset(...) for the
benchmark data. That's redundant tokenization/download work across 4
ranks, but harmless at this scale -- unlike the 120GB pretraining corpus
(why ShardedTextDataset in train.py bothers with an on-disk index at all).

PREFINETUNE: arabic_squad / polyglot_ner_ar are prefinetune resources,
not benchmark tasks -- run them first (make prefinetune-arcd /
prefinetune-anercorp) and point finetune_arcd.yaml / finetune_anercorp.yaml's
pretrained_ckpt at the resulting best.pt, before running the real
arcd/anercorp benchmark. See data/finetune_datasets.py's module docstring.
"""

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

    # 50/50 ARCD Benchmark strategy
    "arcd_50_50": {"loader": load_arcd_50_50, "task_type": "qa"},

    # Joint Pre-finetuning (Arabic-SQuAD + 50% ARCD)
    "squad_plus_arcd50": {"loader": load_squad_plus_arcd50, "task_type": "qa"},
    "squad_plus_tydiqa_ar": {"loader": load_squad_plus_tydiqa_ar, "task_type": "qa"},

    # Prefinetune-only resources
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
                ctx_end = real_len - 2  # exclude trailing [SEP]
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
        print(f"loading dataset for task={task_name} (world_size={world_size}) ...")
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
            print(f"loaded pretrained encoder from {cfg['pretrained_ckpt']}")
    elif is_main:
        print(
            "WARNING: no pretrained_ckpt set in config -- training encoder from "
            "scratch, this defeats the point of the benchmark"
        )

    # Wrap AFTER loading pretrained weights into the plain module.
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

    # dev/test: only rank 0 evaluates -- same reasoning as train.py's
    # val_loader, so no DistributedSampler needed here.
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
        print(f"world_size={world_size}  effective_batch_size={effective_batch_size}")
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
            dist.barrier()  # let all ranks finish the epoch before rank-0-only eval

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

            print(f"[epoch {epoch + 1}] dev metrics: {metrics}")
            if metrics[primary_metric] > best_metric:
                best_metric = metrics[primary_metric]
                best_state = {
                    k: v.cpu().clone() for k, v in core_model.state_dict().items()
                }
                print(
                    f"  -> new best ({primary_metric}={best_metric:.4f}), checkpointing in memory"
                )

        if ddp:
            dist.barrier()  # rank 0's eval finishes before others start the next epoch

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
        print(f"wrote {result_path}")

        if best_state is not None:
            ckpt_path = os.path.join(cfg["paths"]["output_dir"], f"{task_name}_best.pt")
            torch.save(best_state, ckpt_path)
            print(f"wrote {ckpt_path}")

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)
