"""Chromosome-based splits and sequence encodings (one-hot for the CNN, k-mers for LR)."""
from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Sequence, Tuple

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
    log.info("\n[3/8] Building chromosome splits...")
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


def balance_splits(pos_df: pd.DataFrame, neg_df: pd.DataFrame, seed: int) -> SplitData:
    """Balance positives and negatives within each split by downsampling the larger class."""
    rng = random.Random(seed)
    data: SplitData = {}
    for split in ("train", "val", "test"):
        pos = pos_df.loc[pos_df["split"] == split, "sequence"].tolist()
        neg = neg_df.loc[neg_df["split"] == split, "sequence"].tolist()
        n = min(len(pos), len(neg))
        rng.shuffle(pos); rng.shuffle(neg)
        combined = [(s, 1) for s in pos[:n]] + [(s, 0) for s in neg[:n]]
        rng.shuffle(combined)
        data[split] = ([s for s, _ in combined], [l for _, l in combined])
        log.info(f"  {split.capitalize():<5} set           : {2*n:,} sequences ({n:,} pos / {n:,} neg)")

    present = sorted(set(pos_df["chrom"]))
    for split, hint in (("test", "split.test_chroms"), ("val", "split.val_chroms"), ("train", "the split")):
        if len(data[split][0]) == 0:
            raise PipelineError(f"The {split} set is empty. Chromosomes in data: {present}. "
                                f"Adjust {hint} to chromosomes present in your data.")
    return data


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
