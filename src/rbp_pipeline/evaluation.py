"""Evaluation metrics on the held-out test set."""
from __future__ import annotations

from typing import Dict, Sequence

from sklearn.metrics import roc_auc_score

from .genome import gc_fraction


def gc_scores(seqs: Sequence[str]):
    """GC fraction per sequence: a one-number baseline 'model' for composition bias."""
    return [gc_fraction(s) for s in seqs]


def auc_table(labels: Sequence[int], scores: Dict[str, Sequence[float]]) -> Dict[str, float]:
    """ROC-AUC for each named score vector."""
    return {name: float(roc_auc_score(labels, s)) for name, s in scores.items()}


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
