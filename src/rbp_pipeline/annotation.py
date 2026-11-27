"""
Strand-aware genomic region annotation (UTR / exonic / intronic / intergenic) with bedtools.

The GTF is converted once into a small sorted feature BED that is cached and reused, then
peaks are intersected on the same strand. On GENCODE this takes minutes, not hours.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Tuple

import pandas as pd

from . import PipelineError
from .genome import VALID_CHROMS

log = logging.getLogger("rbp_pipeline")

# feature type in GTF -> (region class, priority). Higher priority wins when a peak overlaps several.
FEATURE_CLASSES = {"UTR": ("UTR", 3), "CDS": ("exonic", 2), "exon": ("exonic", 2), "gene": ("intronic", 1)}
_C_LOCALE = {**os.environ, "LC_ALL": "C"}
_GENE_NAME = re.compile(r'gene_name "([^"]+)"')


def feature_bed_from_gtf(gtf: Path, cache_dir: Path) -> Path:
    """Build (or reuse) a deduplicated, sorted BED of GTF features: chrom start end gene class strand."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / (gtf.name + ".features.bed")
    if cached.exists() and cached.stat().st_mtime >= gtf.stat().st_mtime:
        log.info(f"  Using cached GTF features: {cached}")
        return cached

    log.info("  Building GTF feature index (one-off, ~1-2 min for GENCODE)...")
    unsorted = cached.with_suffix(".unsorted")
    with open(gtf) as fin, open(unsorted, "w") as fout:
        for line in fin:
            if line.startswith("#"):
                continue
            cols = line.split("\t", 8)
            if len(cols) < 9 or cols[2] not in FEATURE_CLASSES or cols[0] not in VALID_CHROMS:
                continue
            m = _GENE_NAME.search(cols[8])
            region, _ = FEATURE_CLASSES[cols[2]]
            fout.write(f"{cols[0]}\t{int(cols[3]) - 1}\t{cols[4]}\t{m.group(1) if m else '.'}\t{region}\t{cols[6]}\n")
    _sort_bed(unsorted, cached, unique=True)
    unsorted.unlink(missing_ok=True)
    return cached


def _sort_bed(src: Path, dst: Path, unique: bool = False) -> None:
    # With -u, sort compares only the key fields, so every column must be a key or lines that
    # merely share chrom+start (e.g. a gene and its 5' UTR) would be dropped as "duplicates".
    keys = ["-k1,1", "-k2,2n"] + (["-k3,3n", "-k4,4", "-k5,5", "-k6,6", "-u"] if unique else [])
    cmd = ["sort"] + keys + ["-o", str(dst), str(src)]
    subprocess.run(cmd, check=True, env=_C_LOCALE)


def annotate_regions(peaks_df: pd.DataFrame, gtf: Path, work_dir: Path,
                     cache_dir: Path) -> pd.DataFrame:
    """
    Add `region_type` and `gene_name` to every peak. Peaks are matched to features on the
    same strand (an antisense gene doesn't make a peak 'intronic'). Unstranded peaks match
    features on either strand.
    """
    features = feature_bed_from_gtf(Path(gtf), cache_dir)
    priority = {cls: p for cls, p in FEATURE_CLASSES.values()}

    peaks_bed = work_dir / "_annot_peaks.bed"
    sorted_bed = work_dir / "_annot_peaks.sorted.bed"
    best = {}
    try:
        for stranded in (True, False):
            subset = peaks_df[(peaks_df["strand"] != ".") if stranded else (peaks_df["strand"] == ".")]
            if subset.empty:
                continue
            subset[["chrom", "start", "end", "peak_id"]].assign(score=0, strand=subset["strand"]) \
                .to_csv(peaks_bed, sep="\t", index=False, header=False)
            _sort_bed(peaks_bed, sorted_bed)
            cmd = ["bedtools", "intersect", "-a", str(sorted_bed), "-b", str(features),
                   "-wa", "-wb", "-sorted"] + (["-s"] if stranded else [])
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, check=True, env=_C_LOCALE).stdout
            except subprocess.CalledProcessError as e:
                raise PipelineError(f"bedtools intersect failed during annotation:\n{e.stderr}") from e
            for line in out.splitlines():
                c = line.split("\t")
                peak_id, gene, cls = c[3], c[9], c[10]
                p = priority[cls]
                if peak_id not in best or p > best[peak_id][0]:
                    best[peak_id] = (p, cls, gene)
    finally:
        peaks_bed.unlink(missing_ok=True)
        sorted_bed.unlink(missing_ok=True)

    out_df = peaks_df.copy()
    out_df["region_type"] = [best.get(pid, (0, "intergenic", "."))[1] for pid in out_df["peak_id"]]
    out_df["gene_name"] = [best.get(pid, (0, "intergenic", "."))[2] for pid in out_df["peak_id"]]

    counts = out_df["region_type"].value_counts()
    log.info("  Region breakdown:")
    for rt, cnt in counts.items():
        log.info(f"    {rt:<15} : {cnt:>6,}  ({100*cnt/len(out_df):.1f}%)")
    return out_df


def skip_annotation(peaks_df: pd.DataFrame, reason: str) -> pd.DataFrame:
    log.info(f"  Skipped ({reason})")
    out = peaks_df.copy()
    out["region_type"] = "not_annotated"
    out["gene_name"] = "."
    return out
