"""
Metrics for the MSA benchmark suite. Kept separate from finetune.py, same
philosophy as src/utils/eval.py being separate from train.py: metric logic
that can be tested or reused independently of the training loop.
"""

import collections
import re
import string

import numpy as np


def classification_metrics(preds, labels):
    """Accuracy + macro-F1, computed by hand (no sklearn dependency --
    same dependency-free spirit as the BloomFilter in prepare_data.py)."""
    preds = np.array(preds)
    labels = np.array(labels)
    acc = (preds == labels).mean()
    classes = sorted(set(labels.tolist()) | set(preds.tolist()))
    f1s = []
    for c in classes:
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return {"accuracy": float(acc), "macro_f1": float(np.mean(f1s))}


def seqeval_ner_metrics(pred_tags, gold_tags):
    """Entity-level precision/recall/F1 over BIO tag sequences -- the
    metric actually reported in the ANERcorp/AraBERT/ARBERT literature,
    NOT token-level accuracy (token accuracy is inflated by the dominant
    'O' class and isn't comparable to published numbers).

    Requires `seqeval` (pip install seqeval --break-system-packages).
    Falls back to token accuracy with a loud warning if it's missing, so
    this never silently reports a number that looks like the standard
    metric but isn't.
    """
    try:
        from seqeval.metrics import f1_score, precision_score, recall_score

        return {
            "precision": precision_score(gold_tags, pred_tags),
            "recall": recall_score(gold_tags, pred_tags),
            "f1": f1_score(gold_tags, pred_tags),
        }
    except ImportError:
        correct = sum(
            p == g
            for pseq, gseq in zip(pred_tags, gold_tags)
            for p, g in zip(pseq, gseq)
        )
        total = sum(len(gseq) for gseq in gold_tags)
        print(
            "[finetune_metrics] seqeval not installed -- reporting TOKEN accuracy, "
            "which is NOT comparable to published entity-level F1. "
            "Run: pip install seqeval --break-system-packages"
        )
        return {"token_accuracy": correct / max(1, total)}


_AR_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")


def _normalize_arabic_answer(s):
    """SQuAD-style normalization adapted for Arabic: strip diacritics,
    punctuation, and collapse whitespace, so e.g. trailing punctuation or
    an optional diacritic doesn't spuriously fail an exact match. Reuses
    the same diacritics regex as prepare_data.py on purpose, for
    consistency across the codebase."""
    s = _AR_DIACRITICS.sub("", s)
    s = "".join(ch for ch in s if ch not in string.punctuation and ch not in "،؛؟”“")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def qa_metrics(preds, golds):
    """preds: {id: predicted_answer_text}
    golds: {id: [gold_answer_text, ...]}   (ARCD/SQuAD allow multiple refs)
    Returns SQuAD-style EM and token-overlap F1, averaged over examples,
    taking the best-matching gold answer per example.
    """
    em_total, f1_total = 0.0, 0.0
    for qid, gold_list in golds.items():
        pred = _normalize_arabic_answer(preds.get(qid, ""))
        best_em, best_f1 = 0.0, 0.0
        for gold in gold_list:
            gold_n = _normalize_arabic_answer(gold)
            em = float(pred == gold_n)
            pred_toks, gold_toks = pred.split(), gold_n.split()
            common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
            num_same = sum(common.values())
            if num_same == 0:
                f1 = 0.0
            else:
                prec = num_same / max(1, len(pred_toks))
                rec = num_same / max(1, len(gold_toks))
                f1 = 2 * prec * rec / (prec + rec)
            best_em, best_f1 = max(best_em, em), max(best_f1, f1)
        em_total += best_em
        f1_total += best_f1
    n = max(1, len(golds))
    return {"exact_match": 100 * em_total / n, "f1": 100 * f1_total / n}
