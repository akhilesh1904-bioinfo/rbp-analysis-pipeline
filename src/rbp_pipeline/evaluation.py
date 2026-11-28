"""Evaluation metrics on the held-out test set: threshold-free, thresholded, calibration, CIs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

from .genome import gc_fraction

# Metrics that need probabilities (not meaningful for the raw GC-fraction baseline)
THRESHOLD_METRICS = ("precision", "recall", "f1", "mcc", "brier")


def gc_scores(seqs: Sequence[str]) -> List[float]:
    """GC fraction per sequence: a one-number baseline 'model' for composition bias."""
    return [gc_fraction(s) for s in seqs]


def auc_table(labels: Sequence[int], scores: Dict[str, Sequence[float]]) -> Dict[str, float]:
    """ROC-AUC for each named score vector."""
    return {name: float(roc_auc_score(labels, s)) for name, s in scores.items()}


def compute_metrics(labels: Sequence[int], scores: Sequence[float], threshold: Optional[float] = 0.5) -> Dict:
    """
    ROC-AUC and PR-AUC (average precision) for any score. If `threshold` is given the scores are
    treated as probabilities and precision, recall, F1, MCC, Brier score and the confusion matrix
    at that threshold are added.
    """
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    out = {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
           "n": int(len(y)), "n_pos": int(y.sum())}
    if threshold is not None:
        pred = (s >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        out.update({
            "threshold": threshold,
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y, pred)),
            "brier": float(brier_score_loss(y, s)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        })
    return out


def _clusters(n: int, groups: Optional[Sequence]) -> List[np.ndarray]:
    """Index arrays, one per resampling unit (each sequence on its own when groups is None)."""
    if groups is None:
        return [np.array([i]) for i in range(n)]
    if len(groups) != n:
        raise ValueError(f"groups has {len(groups)} entries for {n} sequences")
    members: Dict[Any, List[int]] = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    return [np.array(v) for v in members.values()]


def _resample(rng: np.random.Generator, clusters: List[np.ndarray]) -> np.ndarray:
    """Draw as many units as there are, with replacement, and return their pooled indices."""
    pick = rng.integers(0, len(clusters), len(clusters))
    return np.concatenate([clusters[i] for i in pick])


def bootstrap_ci(labels: Sequence[int], scores: Sequence[float], metric: str = "roc_auc",
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
                 groups: Optional[Sequence] = None) -> Tuple[float, float]:
    """
    Percentile bootstrap CI. Without `groups`, test sequences are resampled one by one. With
    `groups` (one label per sequence), whole groups are resampled: e.g. a peak together with
    its matched negative, or all pairs from one gene, so dependent sequences stay together.
    """
    if n_boot <= 0:
        return (float("nan"), float("nan"))
    fn = roc_auc_score if metric == "roc_auc" else average_precision_score
    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    clusters = _clusters(len(y), groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = _resample(rng, clusters)
        if y[idx].min() == y[idx].max():
            continue
        vals.append(fn(y[idx], s[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_diff(labels: Sequence[int], a: Sequence[float], b: Sequence[float],
                          n_boot: int = 1000, seed: int = 0,
                          groups: Optional[Sequence] = None) -> Dict[str, float]:
    """ROC-AUC(a) - ROC-AUC(b) on the same test set, with a paired (optionally grouped) bootstrap 95% CI."""
    y, a, b = np.asarray(labels), np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    if n_boot <= 0:
        return {"diff": diff, "ci_low": float("nan"), "ci_high": float("nan")}
    clusters = _clusters(len(y), groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = _resample(rng, clusters)
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"diff": diff, "ci_low": float(lo), "ci_high": float(hi)}


def summarise_seeds(per_seed: List[Dict[str, Dict]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """[{model: metrics}, ...] -> {model: {metric: {mean, sd, values}}} for scalar metrics."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model in per_seed[0]:
        out[model] = {}
        for metric, v in per_seed[0][model].items():
            if not isinstance(v, (int, float)) or metric in ("n", "n_pos", "threshold"):
                continue
            vals = [run[model][metric] for run in per_seed]
            out[model][metric] = {"mean": float(np.mean(vals)),
                                  "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                  "values": [float(x) for x in vals]}
    return out


def interpret_auc(auc: float) -> str:
    if auc >= 0.85:
        return (f"Strong model performance (AUC = {auc:.3f}). The model reliably distinguishes binding from "
                "non-binding sites, indicating clear sequence-level features driving binding.")
    if auc >= 0.70:
        return (f"Good model performance (AUC = {auc:.3f}). The model captures meaningful binding sequence "
                "features, though some binding sites may lack strong motif signal.")
    if auc >= 0.55:
        return (f"Moderate model performance (AUC = {auc:.3f}). The model performs above chance but binding "
                "may be driven by context beyond short sequence motifs.")
    return (f"Low model performance (AUC = {auc:.3f}). Binding sites are not well-distinguished from "
            "matched negatives by sequence features alone.")
