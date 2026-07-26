"""
Aggregates the per-seed result JSONs written by finetune.py
(results_finetune/<task>/<task>_seed<seed>.json) into a mean +- std
summary per task, printed as a markdown table ready to paste into the
paper. Kept as a standalone script rather than folded into finetune.py
since aggregation happens once you have several runs, not per-run.

Usage:
  python src/utils/aggregate_results.py --results_dir results_finetune
"""

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict


def load_results(results_dir):
    """Returns {task_name: [result_dict, ...]} across all seeds found."""
    by_task = defaultdict(list)
    for path in glob.glob(os.path.join(results_dir, "*", "*_seed*.json")):
        with open(path) as f:
            r = json.load(f)
        by_task[r["task"]].append(r)
    return by_task


def summarize(results):
    """results: list of {"test": {metric: value, ...}, "seed": int, ...}
    Returns {metric: (mean, std, n)} across seeds. std is 0.0 (not NaN)
    for n==1 so a single-seed run still prints cleanly -- but the table
    caller should flag n==1 as "not yet a real variance estimate"."""
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
        print(f"no result JSONs found under {results_dir}/*/*_seed*.json")
        return

    print("| Task | Metric | Mean | Std | N seeds |")
    print("|---|---|---|---|---|")
    for task, results in sorted(by_task.items()):
        summary = summarize(results)
        for metric, (mean, std, n) in sorted(summary.items()):
            flag = "  (single run, not a variance estimate)" if n == 1 else ""
            print(f"| {task} | {metric} | {mean:.4f} | {std:.4f} | {n}{flag} |")

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
