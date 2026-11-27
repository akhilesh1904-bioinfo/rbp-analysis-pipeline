"""
Sequence-level leakage checks between the training set and the validation / test sets.

Chromosome splitting prevents the same genomic interval appearing in two splits, but
repeats, paralogs and duplicated UTRs can put (near-)identical sequences on different
chromosomes. A test sequence that is identical to, or shares a long exact substring with,
a training sequence lets a model score well by memorisation.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, Iterable, Set

import pandas as pd

from .genome import revcomp

log = logging.getLogger("rbp_pipeline")


def _informative(kmer: str) -> bool:
    """Skip low-complexity k-mers (e.g. poly-A, AT repeats) that match by chance in any genome."""
    if "N" in kmer:
        return False
    counts = Counter(kmer)
    return len(counts) >= 3 and max(counts.values()) <= 0.5 * len(kmer)


def kmer_set(seqs: Iterable[str], k: int) -> Set[str]:
    out: Set[str] = set()
    for s in seqs:
        for i in range(len(s) - k + 1):
            km = s[i:i + k]
            if _informative(km):
                out.add(km)
    return out


def find_leaks(pos_df: pd.DataFrame, neg_df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """
    Flag every val/test sequence (positive or negative) that matches the training set:
      exact     — identical to a training sequence
      revcomp   — identical to the reverse complement of a training sequence
      shared_kmer — shares at least one informative k-mer (either orientation) with training
    Returns one row per flagged sequence: id, split, label, reason.
    """
    both = pd.concat([pos_df.assign(label=1, seq_id=pos_df["peak_id"]),
                      neg_df.assign(label=0, seq_id=neg_df["neg_id"])], ignore_index=True)
    train = both[both["split"] == "train"]["sequence"]
    train_exact = set(train)
    train_rc = {revcomp(s) for s in train}
    train_kmers = kmer_set(train, k)

    rows = []
    for r in both[both["split"] != "train"].itertuples():
        s = r.sequence
        if s in train_exact:
            reason = "exact"
        elif s in train_rc:
            reason = "revcomp"
        else:
            rc = revcomp(s)
            hit = any(s[i:i + k] in train_kmers or rc[i:i + k] in train_kmers
                      for i in range(len(s) - k + 1))
            reason = "shared_kmer" if hit else None
        if reason:
            rows.append({"seq_id": r.seq_id, "split": r.split, "label": r.label, "reason": reason})
    return pd.DataFrame(rows, columns=["seq_id", "split", "label", "reason"])


def summarise_leaks(leaks: pd.DataFrame, pos_df: pd.DataFrame, neg_df: pd.DataFrame, k: int) -> Dict:
    summary: Dict = {"k": k}
    for split in ("val", "test"):
        n_total = int((pos_df["split"] == split).sum() + (neg_df["split"] == split).sum())
        sub = leaks[leaks["split"] == split]
        summary[split] = {
            "n_sequences": n_total, "n_flagged": int(len(sub)),
            "fraction_flagged": (len(sub) / n_total) if n_total else 0.0,
            "by_reason": {k_: int(v) for k_, v in sub["reason"].value_counts().items()},
            "positives_flagged": int((sub["label"] == 1).sum()),
            "negatives_flagged": int((sub["label"] == 0).sum()),
        }
    return summary


def check_leakage(pos_df: pd.DataFrame, neg_df: pd.DataFrame, mode: str, k: int):
    """
    Run the check, log a summary, and (mode == 'remove') drop flagged val/test sequences.
    Returns (pos_df, neg_df, summary, leaks_table).
    """
    log.info(f"  Leakage check       : exact, reverse-complement and shared {k}-mer matches vs training")
    leaks = find_leaks(pos_df, neg_df, k)
    summary = summarise_leaks(leaks, pos_df, neg_df, k)
    for split in ("val", "test"):
        s = summary[split]
        reasons = ", ".join(f"{r} {n}" for r, n in s["by_reason"].items()) or "none"
        log.info(f"    {split:<4}: {s['n_flagged']:,} / {s['n_sequences']:,} sequences flagged "
                 f"({100*s['fraction_flagged']:.1f}%; {reasons}; "
                 f"{s['positives_flagged']} pos / {s['negatives_flagged']} neg)")
    summary["mode"] = mode
    if mode == "remove" and len(leaks):
        flagged = set(leaks["seq_id"])
        pos_df = pos_df[~pos_df["peak_id"].isin(flagged)].reset_index(drop=True)
        neg_df = neg_df[~neg_df["neg_id"].isin(flagged)].reset_index(drop=True)
        log.info(f"    Removed {len(flagged):,} flagged val/test sequences before balancing")
    elif len(leaks) and summary["test"]["fraction_flagged"] > 0.05:
        log.warning("    WARNING: >5% of test sequences resemble training sequences. Consider "
                    "split.leakage: remove and compare results.")
    return pos_df, neg_df, summary, leaks
