"""Attach strand-aware (RNA-sense) sequences to peak and negative tables."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from . import PipelineError
from .genome import Genome, clean_seq, region_key, revcomp

log = logging.getLogger("rbp_pipeline")

MAX_N_FRACTION = 0.10   # sequences with more ambiguous bases than this are dropped


def attach_sequences(df: pd.DataFrame, fetched: Dict[str, str], id_col: str,
                     out_fasta: Path, max_n_fraction: float = MAX_N_FRACTION) -> pd.DataFrame:
    """
    Add a 'sequence' column (RNA sense: reverse-complemented for '-' strand), keyed by
    coordinates so rows can never be misaligned. Drops rows whose sequence could not be
    fetched or is mostly ambiguous bases, and writes a FASTA with id::region(strand) headers.
    """
    seqs = []
    for r in df.itertuples():
        s = clean_seq(fetched.get(region_key(r.chrom, r.start, r.end), ""))
        if r.strand == "-":
            s = revcomp(s)
        seqs.append(s)
    out = df.copy()
    out["sequence"] = seqs
    n_frac = out["sequence"].map(lambda s: s.count("N") / len(s) if s else 1.0)
    keep = n_frac <= max_n_fraction
    dropped = int((~keep).sum())
    if dropped:
        log.info(f"  Dropped {dropped:,} regions (missing from genome or >{max_n_fraction:.0%} N)")
    out = out[keep].reset_index(drop=True)
    write_fasta(out, id_col, out_fasta)
    return out


def write_fasta(df: pd.DataFrame, id_col: str, path: Path) -> None:
    with open(path, "w") as f:
        for r in df.itertuples():
            f.write(f">{getattr(r, id_col)}::{region_key(r.chrom, r.start, r.end)}({r.strand})\n"
                    f"{r.sequence}\n")


def extract_sequences(pos_df: pd.DataFrame, neg_df: pd.DataFrame, genome: Genome,
                      out_dir: Path, max_len: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log.info("\n[3/8] Extracting strand-aware sequences...")
    regions = pd.concat([pos_df[["chrom", "start", "end"]], neg_df[["chrom", "start", "end"]]])
    fetched = genome.fetch(regions, out_dir)

    pos_df = attach_sequences(pos_df, fetched, "peak_id", out_dir / "positives.fa")
    neg_df = attach_sequences(neg_df, fetched, "neg_id", out_dir / "negatives.fa")

    log.info(f"  Positive sequences  : {len(pos_df):,}")
    log.info(f"  Negative sequences  : {len(neg_df):,}")
    if len(pos_df):
        lens = pos_df["sequence"].str.len()
        log.info(f"  Length range        : {lens.min()}–{lens.max()} bp  (median {int(lens.median())} bp)")
        n_long = int((lens > max_len).sum())
        if n_long:
            log.info(f"  Note: {n_long:,} positives longer than model.max_len {max_len} bp "
                     "will be centre-cropped for the CNN")
    if len(pos_df) == 0 or len(neg_df) == 0:
        raise PipelineError("No sequences extracted — check that the genome FASTA matches your BED "
                            "(GRCh38, chr-prefixed names).")
    return pos_df, neg_df
