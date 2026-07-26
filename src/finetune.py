"""
Single-GPU finetuning + evaluation driver for the MSA benchmark suite
(see docs/BENCHMARKING.md). Unlike train.py this is NOT a DDP script --
these benchmark datasets (hundreds to ~90k rows) comfortably finetune on
one RTX 2080Ti in well under an hour, so DDP would only add complexity
with no real payoff at this scale. If you want to burn all 4 GPUs, just
launch 4 different tasks/seeds in parallel instead (see BENCHMARKING.md).

Launch:
  python src/finetune.py --config configs/finetune_hard.yaml
  python src/finetune.py --config configs/finetune_xnli.yaml
  python src/finetune.py --config configs/finetune_anercorp.yaml
  python src/finetune.py --config configs/finetune_arcd.yaml
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from data.finetune_datasets import (
    collate_classification,
    collate_ner,
    collate_qa,
    load_anercorp,
    load_arcd,
    load_hard,
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
}


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


def evaluate_qa(model, loader, device, fp16):
    model.eval()
    preds, golds = {}, {}
    with torch.no_grad():
        for input_ids, attn_mask, starts, ends, ids, gold in loader:
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
            with torch.amp.autocast(device_type="cuda", enabled=fp16):
                out = model(input_ids, attn_mask)
            start_idx = out["start_logits"].argmax(-1).cpu().tolist()
            end_idx = out["end_logits"].argmax(-1).cpu().tolist()
            for i, qid in enumerate(ids):
                s, e = start_idx[i], end_idx[i]
                if e < s:
                    e = s
                span_ids = input_ids[i, s : e + 1].cpu().tolist()
                # decoding here needs the tokenizer; caller reattaches it
                preds[qid] = span_ids
                golds[qid] = gold[i]
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

    set_seed(cfg["training"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg["data"]["tokenizer_name"])
    assert tokenizer.is_fast, "finetune.py requires a fast tokenizer (word_ids/offsets)"

    print(f"loading dataset for task={task_name} ...")
    if task_type == "ner":
        train_ds, dev_ds, test_ds = task_info["loader"](
            tokenizer, cfg["data"]["max_seq_length"]
        )
        label2id = train_ds.label2id
        id2label = {v: k for k, v in label2id.items()}
        num_labels = len(label2id)
    else:
        train_ds, dev_ds, test_ds = task_info["loader"](
            tokenizer, cfg["data"]["max_seq_length"]
        )
        num_labels = task_info.get("num_labels")

    mcfg = dict(cfg["model"])
    mcfg["vocab_size"] = tokenizer.vocab_size
    mcfg["pad_token_id"] = tokenizer.pad_token_id
    model = build_model(task_type, mcfg, num_labels=num_labels).to(device)

    if cfg.get("pretrained_ckpt"):
        load_pretrained_encoder(model.encoder, cfg["pretrained_ckpt"], device=device)
        print(f"loaded pretrained encoder from {cfg['pretrained_ckpt']}")
    else:
        print(
            "WARNING: no pretrained_ckpt set in config -- training encoder from scratch, "
            "this defeats the point of the benchmark (comparing pretraining quality)"
        )

    tcfg = cfg["training"]
    if task_type == "classification":
        collate = lambda b: collate_classification(b, tokenizer.pad_token_id)
    elif task_type == "ner":
        collate = lambda b: collate_ner(b, tokenizer.pad_token_id)
    else:
        collate = lambda b: collate_qa(b, tokenizer.pad_token_id)

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
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda step: max(0.0, (total_steps - step) / max(1, total_steps))
    )

    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    best_metric, best_state = -1.0, None
    primary_metric = {"classification": "macro_f1", "ner": "f1", "qa": "f1"}[task_type]

    model.train()
    for epoch in range(tcfg["epochs"]):
        pbar = tqdm(
            train_loader, desc=f"epoch {epoch+1}/{tcfg['epochs']}", dynamic_ncols=True
        )
        for batch in pbar:
            optim.zero_grad()
            if task_type == "classification":
                input_ids, attn_mask, labels = batch
                input_ids, attn_mask, labels = (
                    input_ids.to(device),
                    attn_mask.to(device),
                    labels.to(device),
                )
                with torch.amp.autocast(device_type="cuda", enabled=tcfg["fp16"]):
                    out = model(input_ids, attn_mask, labels)
            elif task_type == "ner":
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
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- end of epoch: evaluate on dev ---
        if task_type == "classification":
            metrics = evaluate_classification(model, dev_loader, device, tcfg["fp16"])
        elif task_type == "ner":
            metrics = evaluate_ner(model, dev_loader, device, tcfg["fp16"], id2label)
        else:
            preds, golds = evaluate_qa(model, dev_loader, device, tcfg["fp16"])
            preds_text = {
                qid: tokenizer.decode(ids, skip_special_tokens=True)
                for qid, ids in preds.items()
            }
            metrics = qa_metrics(preds_text, golds)

        print(f"[epoch {epoch+1}] dev metrics: {metrics}")
        if metrics[primary_metric] > best_metric:
            best_metric = metrics[primary_metric]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"  -> new best ({primary_metric}={best_metric:.4f}), checkpointing in memory"
            )

    # --- final test evaluation with best dev checkpoint ---
    if best_state is not None:
        model.load_state_dict(best_state)
    if task_type == "classification":
        test_metrics = evaluate_classification(model, test_loader, device, tcfg["fp16"])
    elif task_type == "ner":
        test_metrics = evaluate_ner(model, test_loader, device, tcfg["fp16"], id2label)
    else:
        preds, golds = evaluate_qa(model, test_loader, device, tcfg["fp16"])
        preds_text = {
            qid: tokenizer.decode(ids, skip_special_tokens=True)
            for qid, ids in preds.items()
        }
        test_metrics = qa_metrics(preds_text, golds)

    print(
        f"=== FINAL test metrics for {task_name} (seed={tcfg.get('seed', 42)}): {test_metrics} ==="
    )

    result_path = os.path.join(
        cfg["paths"]["output_dir"], f"{task_name}_seed{tcfg.get('seed', 42)}.json"
    )
    with open(result_path, "w") as f:
        json.dump(
            {
                "task": task_name,
                "seed": tcfg.get("seed", 42),
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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)
