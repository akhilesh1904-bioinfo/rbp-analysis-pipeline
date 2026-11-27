"""Negative (non-binding) region generation: length-, chromosome- and optionally GC-matched."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from . import PipelineError
from .genome import GRCH38_SIZES, VALID_CHROMS, Genome, gc_fraction

log = logging.getLogger("rbp_pipeline")


def generate_candidates(peaks_bed: Path, genome: Genome, regions_bed: Path,
                        out_bed: Path, rounds: int, seed: int) -> Path:
    """
    Shuffle every peak `rounds` times within `regions_bed` on its own chromosome. Each round gives
    one length- and chromosome-matched region per peak; column 4 keeps the source peak_id.

    bedtools' own -chrom flag is silently ignored when combined with -incl (v2.31), so we
    shuffle one chromosome at a time instead.
    """
    label = "candidate" if rounds > 1 else "matched negative"
    log.info(f"\n[2/8] Generating {label} regions ({rounds} per peak)...")
    work = out_bed.parent

    chrom_sizes = {c: s for c, s in genome.chrom_sizes().items() if c in VALID_CHROMS}
    if not chrom_sizes:
        log.info("  Note: genome .fai unavailable — using standard GRCh38 chromosome sizes")
        chrom_sizes = GRCH38_SIZES

    peak_lines: Dict[str, List[str]] = {}
    with open(peaks_bed) as f:
        for line in f:
            peak_lines.setdefault(line.split("\t", 1)[0], []).append(line)
    region_lines: Dict[str, List[str]] = {}
    with open(regions_bed) as f:
        for line in f:
            if line.strip() and not line.startswith(("#", "track", "browser")):
                chrom = line.split("\t", 1)[0]
                if chrom in peak_lines:
                    region_lines.setdefault(chrom, []).append(line)

    genome_file, chrom_peaks, chrom_regions = (work / "_genome.txt", work / "_shuffle_peaks.bed",
                                               work / "_shuffle_regions.bed")
    outputs, skipped = [], []
    try:
        for i, chrom in enumerate(sorted(peak_lines)):
            if chrom not in chrom_sizes or chrom not in region_lines:
                skipped.append(chrom)
                continue
            genome_file.write_text(f"{chrom}\t{chrom_sizes[chrom]}\n")
            chrom_peaks.write_text("".join(peak_lines[chrom]))
            chrom_regions.write_text("".join(region_lines[chrom]))
            for r in range(rounds):
                cmd = ["bedtools", "shuffle", "-i", str(chrom_peaks), "-g", str(genome_file),
                       "-incl", str(chrom_regions), "-excl", str(peaks_bed), "-noOverlapping",
                       "-maxTries", "1000", "-seed", str(seed + 1000 * r + i)]
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise PipelineError(f"bedtools shuffle failed on {chrom}:\n{e.stderr}") from e
                except FileNotFoundError as e:
                    raise PipelineError("bedtools not found. Install with: conda install -c bioconda bedtools") from e
                outputs.append(result.stdout)
    finally:
        for p in (genome_file, chrom_peaks, chrom_regions):
            p.unlink(missing_ok=True)

    if skipped:
        log.warning(f"  WARNING: no regions / size for {skipped}; no negatives generated there")
    out_bed.write_text("".join(outputs))
    n = sum(len([l for l in o.splitlines() if l.strip()]) for o in outputs)
    log.info(f"  Regions generated   : {n:,} (length- and chromosome-matched)")
    return out_bed


def select_gc_matched(pos_df: pd.DataFrame, cand_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each peak, keep the candidate (same length and chromosome, by construction) whose
    GC content is closest to the peak's. Each genomic interval is used at most once.
    """
    pos_gc = pos_df.set_index("peak_id")["sequence"].map(gc_fraction)
    cand = cand_df[cand_df["source_peak"].isin(pos_gc.index)].copy()
    cand["gc"] = cand["sequence"].map(gc_fraction)
    cand["gc_diff"] = (cand["gc"] - cand["source_peak"].map(pos_gc)).abs()
    cand = cand.sort_values(["gc_diff", "neg_id"])

    chosen, used_peaks, used_regions = [], set(), set()
    for r in cand.itertuples():
        key = (r.chrom, r.start, r.end)
        if r.source_peak in used_peaks or key in used_regions:
            continue
        chosen.append(r.Index)
        used_peaks.add(r.source_peak)
        used_regions.add(key)

    out = cand.loc[chosen].sort_values("neg_id").drop(columns=["gc", "gc_diff"]).reset_index(drop=True)
    n_missing = len(pos_gc) - len(out)
    if n_missing:
        log.info(f"  Note: {n_missing:,} peaks had no usable GC-matched candidate")
    return out


def gc_summary(pos_df: pd.DataFrame, neg_df: pd.DataFrame) -> Tuple[float, float]:
    pos_gc = 100 * pos_df["sequence"].map(gc_fraction).mean()
    neg_gc = 100 * neg_df["sequence"].map(gc_fraction).mean()
    log.info(f"  Mean GC             : positives {pos_gc:.1f}%  |  negatives {neg_gc:.1f}%  "
             f"(difference {pos_gc - neg_gc:+.1f} points)")
    return pos_gc, neg_gc
