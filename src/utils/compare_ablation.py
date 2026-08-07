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


def main(pretrained_dir, ablation_dir):
    pretrained = load_results(pretrained_dir)
    ablation = load_results(ablation_dir)

    tasks = sorted(set(pretrained.keys()) & set(ablation.keys()))
    if not tasks:
        print(
            f"no overlapping tasks found between {pretrained_dir} and {ablation_dir} "
            "-- did both finish running?"
        )
        return

    print(
        "| Task | Metric | Pretrained (mean ± std, n) | No-pretrain (mean ± std, n) | Delta |"
    )
    print("|---|---|---|---|---|")
    for task in tasks:
        pre_summary = summarize(pretrained[task])
        abl_summary = summarize(ablation[task])
        metrics = sorted(set(pre_summary.keys()) & set(abl_summary.keys()))
        for metric in metrics:
            pre_mean, pre_std, pre_n = pre_summary[metric]
            abl_mean, abl_std, abl_n = abl_summary[metric]
            delta = pre_mean - abl_mean
            print(
                f"| {task} | {metric} | {pre_mean:.2f} ± {pre_std:.2f} (n={pre_n}) "
                f"| {abl_mean:.2f} ± {abl_std:.2f} (n={abl_n}) | {delta:+.2f} |"
            )

    missing_pretrained = set(ablation.keys()) - set(pretrained.keys())
    missing_ablation = set(pretrained.keys()) - set(ablation.keys())
    if missing_pretrained:
        print(
            f"\nNote: ablation results found for {sorted(missing_pretrained)} but no matching pretrained results."
        )
    if missing_ablation:
        print(
            f"\nNote: pretrained results found for {sorted(missing_ablation)} but no matching ablation results."
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compare pretrained-encoder fine-tuning results against the "
        "no-pretraining (random-init encoder) ablation."
    )
    p.add_argument("--pretrained_dir", default="results_finetune")
    p.add_argument("--ablation_dir", default="results_finetune/ablation_nopretrain")
    args = p.parse_args()
    main(args.pretrained_dir, args.ablation_dir)
