"""Chromosome-based splits and sequence encodings (one-hot for the CNN, k-mers for LR)."""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import PipelineError
from .genome import clean_seq

log = logging.getLogger("rbp_pipeline")

SplitData = Dict[str, Tuple[List[str], List[int]]]


def assign_split(chrom: str, test_set: set, val_set: set) -> str:
    if chrom in test_set:
        return "test"
    if chrom in val_set:
        return "val"
    return "train"


def assign_splits(pos_df: pd.DataFrame, neg_df: pd.DataFrame, test_chroms: Sequence[str],
                  val_chroms: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add a 'split' column (train / val / test) to positives and negatives by chromosome."""
    log.info("\n[3/9] Building chromosome splits...")
    test_set, val_set = set(test_chroms), set(val_chroms)
    if test_set & val_set:
        raise PipelineError(f"Chromosomes in both test and val sets: {sorted(test_set & val_set)}")
    present = set(pos_df["chrom"])
    for label, chroms in (("test", test_set), ("val", val_set)):
        missing = sorted(chroms - present)
        if missing:
            log.info(f"  Note: {label} chromosomes with no peaks: {missing}")
    pos_df = pos_df.copy()
    neg_df = neg_df.copy()
    pos_df["split"] = [assign_split(c, test_set, val_set) for c in pos_df["chrom"]]
    neg_df["split"] = [assign_split(c, test_set, val_set) for c in neg_df["chrom"]]
    return pos_df, neg_df


BALANCE_MODES = ("paired", "random")
_NO_GENE = {"", ".", "intergenic", "none", "nan"}


def _check_not_empty(data: SplitData, pos_df: pd.DataFrame) -> None:
    present = sorted(set(pos_df["chrom"]))
    for split, hint in (("test", "split.test_chroms"), ("val", "split.val_chroms"), ("train", "the split")):
        if len(data[split][0]) == 0:
            raise PipelineError(f"The {split} set is empty. Chromosomes in data: {present}. "
                                f"Adjust {hint} to chromosomes present in your data.")


def can_pair(pos_df: pd.DataFrame, neg_df: pd.DataFrame) -> bool:
    """True when negatives record the peak they were matched to (column source_peak)."""
    if "source_peak" not in neg_df or "peak_id" not in pos_df:
        return False
    return bool(neg_df["source_peak"].isin(set(pos_df["peak_id"])).any())


def _gene_group(peak_id: str, gene_of: Optional[Dict[str, Any]]) -> str:
    g = (gene_of or {}).get(peak_id)
    if g is None or (isinstance(g, float) and np.isnan(g)) or str(g).strip().lower() in _NO_GENE:
        return f"peak:{peak_id}"
    return f"gene:{g}"


def balance_splits_grouped(pos_df: pd.DataFrame, neg_df: pd.DataFrame, seed: int, mode: str = "paired",
                           gene_of: Optional[Dict[str, Any]] = None
                           ) -> Tuple[SplitData, Dict[str, Dict[str, List[str]]], Dict[str, Any]]:
    """
    Balance each split and return, alongside the data, bootstrap group labels for every sequence.

    mode "paired": keep each positive together with the negative matched to it (negatives'
    source_peak). Positives without a negative in the same split (no usable candidate, or its
    negative removed by the leakage check) and negatives whose peak is gone are dropped, so the
    test set is made of complete matched pairs. Falls back to "random" when negatives carry no
    usable source_peak (e.g. a provided negatives BED).
    mode "random": downsample the larger class at random (the behaviour before v0.6.0).

    Returns (data, groups, summary). groups[split] = {"pair": [...], "gene": [...]} aligned with
    data[split]; "pair" is the source peak, "gene" the source peak's gene (or the peak when it has none).
    """
    if mode not in BALANCE_MODES:
        raise PipelineError(f"split.balance must be one of {BALANCE_MODES}, got '{mode}'")
    if mode == "paired" and not can_pair(pos_df, neg_df):
        log.info("  Note: negatives do not record their source peak; balancing at random instead of by pair")
        mode = "random"

    rng = random.Random(seed)
    data: SplitData = {}
    groups: Dict[str, Dict[str, List[str]]] = {}
    summary: Dict[str, Any] = {"mode": mode}
    for split in ("train", "val", "test"):
        pos = pos_df[pos_df["split"] == split]
        neg = neg_df[neg_df["split"] == split]
        if mode == "paired":
            neg_m = neg[neg["source_peak"].isin(set(pos["peak_id"]))].drop_duplicates("source_peak")
            seq_of = dict(zip(pos["peak_id"], pos["sequence"]))
            items = []
            for pid, nseq in zip(neg_m["source_peak"], neg_m["sequence"]):
                items.append((seq_of[pid], 1, pid))
                items.append((nseq, 0, pid))
            n = len(neg_m)
        else:
            p_seqs, n_seqs = pos["sequence"].tolist(), neg["sequence"].tolist()
            n = min(len(p_seqs), len(n_seqs))
            rng.shuffle(p_seqs); rng.shuffle(n_seqs)
            items = [(sq, 1, f"seq:{split}:p{i}") for i, sq in enumerate(p_seqs[:n])] + \
                    [(sq, 0, f"seq:{split}:n{i}") for i, sq in enumerate(n_seqs[:n])]
        rng.shuffle(items)
        data[split] = ([sq for sq, _, _ in items], [lab for _, lab, _ in items])
        pair = [g for _, _, g in items]
        gene = [_gene_group(g, gene_of) if mode == "paired" else g for g in pair]
        groups[split] = {"pair": pair, "gene": gene}
        summary[split] = {"positives_in": int(len(pos)), "negatives_in": int(len(neg)), "kept_pairs": int(n),
                          "positives_dropped": int(len(pos) - n), "negatives_dropped": int(len(neg) - n),
                          "n_genes": int(len(set(gene)))}
        extra = ""
        if len(pos) - n or len(neg) - n:
            extra = f"  (dropped {len(pos) - n:,} pos / {len(neg) - n:,} neg without a {'partner' if mode == 'paired' else 'balance'})"
        log.info(f"  {split.capitalize():<5} set           : {2*n:,} sequences ({n:,} pos / {n:,} neg){extra}")

    _check_not_empty(data, pos_df)
    return data, groups, summary


def balance_splits(pos_df: pd.DataFrame, neg_df: pd.DataFrame, seed: int, mode: str = "random") -> SplitData:
    """Balance positives and negatives within each split (see balance_splits_grouped)."""
    return balance_splits_grouped(pos_df, neg_df, seed, mode)[0]


def build_splits(pos_df: pd.DataFrame, neg_df: pd.DataFrame, test_chroms: Sequence[str],
                 val_chroms: Sequence[str], seed: int) -> Tuple[SplitData, pd.DataFrame, pd.DataFrame]:
    """
    Split positives and negatives by chromosome, then balance each split by downsampling.
    Returns ({split: (seqs, labels)}, pos_df with 'split', neg_df with 'split').
    """
    pos_df, neg_df = assign_splits(pos_df, neg_df, test_chroms, val_chroms)
    return balance_splits(pos_df, neg_df, seed), pos_df, neg_df


# ─────────────────────────────────────────────────────────────────────────────
# Encodings
# ─────────────────────────────────────────────────────────────────────────────

_BASE_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


def fit_length(seq: str, length: int) -> str:
    """Centre-crop sequences longer than `length` (binding is usually near the peak centre)."""
    if len(seq) <= length:
        return seq
    offset = (len(seq) - length) // 2
    return seq[offset:offset + length]


def one_hot(seq: str, max_len: int) -> np.ndarray:
    """4 x max_len one-hot matrix (A, C, G, T rows). N and padding stay all-zero."""
    arr = np.zeros((4, max_len), dtype=np.float32)
    for i, b in enumerate(fit_length(seq, max_len)):
        idx = _BASE_INDEX.get(b)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr


class SeqDataset(Dataset):
    def __init__(self, seqs: Sequence[str], labels: Optional[Sequence[int]] = None, max_len: int = 126):
        self.seqs = seqs
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = torch.tensor(one_hot(self.seqs[idx], self.max_len))
        if self.labels is not None:
            return x, torch.tensor(self.labels[idx], dtype=torch.float32)
        return x


def seq_to_kmers(seq: str, ks=(2, 3)) -> str:
    """Space-separated k-mer 'words' for CountVectorizer; k-mers containing N are skipped."""
    seq = clean_seq(seq)
    tokens = []
    for k in ks:
        tokens.extend(kmer for kmer in (seq[i:i + k] for i in range(len(seq) - k + 1)) if "N" not in kmer)
    return " ".join(tokens)
