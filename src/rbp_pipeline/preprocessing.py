"""Reading and validating peak / region BED files."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from . import PipelineError
from .genome import VALID_CHROMS, normalise_chrom

log = logging.getLogger("rbp_pipeline")

_HEADER_PREFIXES = ("#", "track", "browser")


def _to_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except ValueError:
        return None


def apply_window(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Resize each interval to `window` bp centred on its midpoint (clipped at 0)."""
    df = df.copy()
    mid = (df["start"] + df["end"]) // 2
    df["start"] = (mid - window // 2).clip(lower=0)
    df["end"] = df["start"] + window
    return df


def validate_bed(bed_path: Path, out_dir: Path,
                 min_log2fc: Optional[float] = None,
                 min_log10p: Optional[float] = None,
                 window: Optional[int] = None) -> pd.DataFrame:
    """
    Parse an ENCODE eCLIP BED, apply optional significance filters and windowing, and
    write `peaks_cleaned.bed` (name column = stable peak_id) to `out_dir`.
    """
    log.info("\n[1/8] Validating BED file...")
    rows = []
    skipped = renamed = filtered = 0
    with open(bed_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith(_HEADER_PREFIXES):
                continue
            cols = line.split("\t")
            if len(cols) < 3:
                skipped += 1; continue
            raw_chrom = cols[0]
            chrom = normalise_chrom(raw_chrom)
            try:
                start, end = int(cols[1]), int(cols[2])
            except ValueError:
                skipped += 1; continue
            if start < 0 or end <= start or chrom not in VALID_CHROMS:
                skipped += 1; continue
            strand = cols[5] if len(cols) > 5 else "."
            if strand not in ("+", "-", "."):
                skipped += 1; continue

            log2fc = _to_float(cols[6]) if len(cols) > 6 else None
            log10p = _to_float(cols[7]) if len(cols) > 7 else None
            if min_log2fc is not None and (log2fc is None or log2fc < min_log2fc):
                filtered += 1; continue
            if min_log10p is not None and (log10p is None or log10p < min_log10p):
                filtered += 1; continue

            if chrom != raw_chrom:
                renamed += 1
            rows.append({
                "chrom": chrom, "start": start, "end": end,
                "name": cols[3] if len(cols) > 3 else ".",
                "score": _to_float(cols[4]) if len(cols) > 4 else 0.0,
                "strand": strand, "log2fc": log2fc, "log10p": log10p,
            })

    if not rows:
        raise PipelineError(
            f"No usable peaks in {bed_path} ({skipped:,} invalid lines, {filtered:,} removed by "
            "significance filters). Expected tab-separated BED with chromosomes chr1-22/X/Y/M (or 1-22/X/Y/MT).")

    df = pd.DataFrame(rows)
    if window is not None:
        df["orig_start"], df["orig_end"] = df["start"], df["end"]
        df = apply_window(df, window)
    before = len(df)
    # Same interval on opposite strands = two different RNA binding events, so keep strand in the key
    df = df.drop_duplicates(subset=["chrom", "start", "end", "strand"]).reset_index(drop=True)
    df["peak_length"] = df["end"] - df["start"]
    df["peak_id"] = [f"peak_{i}" for i in range(len(df))]

    log.info(f"  Peaks loaded        : {len(df):,}")
    log.info(f"  Skipped (invalid)   : {skipped:,}")
    if min_log2fc is not None or min_log10p is not None:
        log.info(f"  Removed by filters  : {filtered:,} (log2FC >= {min_log2fc}, -log10p >= {min_log10p})")
    if renamed:
        log.info(f"  Chrom names fixed   : {renamed:,} (e.g. '1' -> 'chr1')")
    if window is not None:
        log.info(f"  Window              : every peak resized to {window} bp around its midpoint")
    log.info(f"  Duplicates removed  : {before - len(df):,}")
    log.info(f"  Chromosomes         : {sorted(df['chrom'].unique())}")
    log.info(f"  Peak length range   : {df['peak_length'].min()}–{df['peak_length'].max()} bp")
    log.info(f"  Median peak length  : {int(df['peak_length'].median())} bp")
    n_unstranded = int((df["strand"] == ".").sum())
    if n_unstranded:
        log.warning(f"  WARNING: {n_unstranded:,} peaks have no strand ('.'); the + strand will be used. "
                    "eCLIP is stranded — check your input.")

    cleaned = Path(out_dir) / "peaks_cleaned.bed"
    out = df[["chrom", "start", "end", "peak_id", "score", "strand"]].copy()
    out["score"] = out["score"].fillna(0)
    out.to_csv(cleaned, sep="\t", index=False, header=False)
    log.info(f"  Cleaned BED saved   : {cleaned}")
    return df


def read_bed(path: Path, id_prefix: str) -> pd.DataFrame:
    """Read a simple BED (>=3 columns) into a DataFrame with a stable `<id_prefix>_id` per row."""
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith(_HEADER_PREFIXES):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3:
                continue
            try:
                start, end = int(cols[1]), int(cols[2])
            except ValueError:
                continue
            chrom = normalise_chrom(cols[0])
            if end <= start or chrom not in VALID_CHROMS:
                continue
            rows.append({"chrom": chrom, "start": start, "end": end,
                         "source_peak": cols[3] if len(cols) > 3 else ".",
                         "strand": cols[5] if len(cols) > 5 and cols[5] in ("+", "-") else "."})
    df = pd.DataFrame(rows, columns=["chrom", "start", "end", "source_peak", "strand"])
    df[f"{id_prefix}_id"] = [f"{id_prefix}_{i}" for i in range(len(df))]
    return df
