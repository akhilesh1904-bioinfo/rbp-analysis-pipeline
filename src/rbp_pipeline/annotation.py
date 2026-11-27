"""
Strand-aware genomic region annotation (UTR / exonic / intronic / intergenic) with bedtools,
and stranded region sets (UTR / exonic / intronic) for drawing negatives.

The GTF is converted once into a small sorted feature BED that is cached and reused, then
peaks are intersected on the same strand. On GENCODE this takes minutes, not hours.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import pandas as pd

from . import PipelineError
from .genome import VALID_CHROMS

log = logging.getLogger("rbp_pipeline")

# feature type in GTF -> (region class, priority). Higher priority wins when a peak overlaps several.
FEATURE_CLASSES = {"UTR": ("UTR", 3), "five_prime_utr": ("UTR", 3), "three_prime_utr": ("UTR", 3),   # GENCODE / Ensembl
                   "CDS": ("exonic", 2), "exon": ("exonic", 2), "gene": ("intronic", 1)}
_C_LOCALE = {**os.environ, "LC_ALL": "C"}
_GENE_NAME = re.compile(r'gene_name "([^"]+)"')
_GENE_TYPE = re.compile(r'gene_(?:type|biotype) "([^"]+)"')
REGION_CLASSES = ("UTR", "exonic", "intronic")
FEATURE_INDEX_VERSION = "v2"     # bump when the cached feature BED layout changes


def feature_bed_from_gtf(gtf: Path, cache_dir: Path) -> Path:
    """
    Build (or reuse) a deduplicated, sorted BED of GTF features:
    chrom start end gene_name class strand gene_type
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{gtf.name}.features.{FEATURE_INDEX_VERSION}.bed"
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
            t = _GENE_TYPE.search(cols[8])
            region, _ = FEATURE_CLASSES[cols[2]]
            fout.write(f"{cols[0]}\t{int(cols[3]) - 1}\t{cols[4]}\t{m.group(1) if m else '.'}\t{region}\t"
                       f"{cols[6]}\t{t.group(1) if t else '.'}\n")
    _sort_bed(unsorted, cached, unique=True)
    unsorted.unlink(missing_ok=True)
    return cached


def _sort_bed(src: Path, dst: Path, unique: bool = False) -> None:
    # With -u, sort compares only the key fields, so every column must be a key or lines that
    # merely share chrom+start (e.g. a gene and its 5' UTR) would be dropped as "duplicates".
    keys = ["-k1,1", "-k2,2n"] + (["-k3,3n", "-k4,4", "-k5,5", "-k6,6", "-k7,7", "-u"] if unique else [])
    cmd = ["sort"] + keys + ["-o", str(dst), str(src)]
    subprocess.run(cmd, check=True, env=_C_LOCALE)


def _intersect(df: pd.DataFrame, id_col: str, features: Path, work_dir: Path, strand_flag: Optional[str]):
    """Yield (id, gene, class, gene_type) for every overlap. strand_flag: '-s', '-S' or None."""
    peaks_bed = work_dir / "_annot_peaks.bed"
    sorted_bed = work_dir / "_annot_peaks.sorted.bed"
    try:
        df[["chrom", "start", "end", id_col]].assign(score=0, strand=df["strand"]) \
            .to_csv(peaks_bed, sep="\t", index=False, header=False)
        _sort_bed(peaks_bed, sorted_bed)
        cmd = ["bedtools", "intersect", "-a", str(sorted_bed), "-b", str(features),
               "-wa", "-wb", "-sorted"] + ([strand_flag] if strand_flag else [])
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=True, env=_C_LOCALE).stdout
        except subprocess.CalledProcessError as e:
            raise PipelineError(f"bedtools intersect failed during annotation:\n{e.stderr}") from e
    finally:
        peaks_bed.unlink(missing_ok=True)
        sorted_bed.unlink(missing_ok=True)
    for line in out.splitlines():
        c = line.split("\t")
        yield c[3], c[9], c[10], (c[12] if len(c) > 12 else ".")


def annotate_regions(peaks_df: pd.DataFrame, gtf: Path, work_dir: Path, cache_dir: Path,
                     id_col: str = "peak_id", quiet: bool = False) -> pd.DataFrame:
    """
    Add `region_type`, `gene_name`, `gene_type` and `antisense` to every region. Regions are matched
    to features on the same strand (an antisense gene doesn't make a peak 'intronic'); `antisense`
    marks stranded regions with no same-strand gene but a gene on the opposite strand. Unstranded
    regions match features on either strand.
    """
    features = feature_bed_from_gtf(Path(gtf), cache_dir)
    priority = {cls: p for cls, p in FEATURE_CLASSES.values()}
    best: Dict[str, Tuple[int, str, str, str]] = {}

    def collect(hits):
        for rid, gene, cls, gtype in hits:
            p = priority[cls]
            if rid not in best or p > best[rid][0]:
                best[rid] = (p, cls, gene, gtype)

    stranded = peaks_df[peaks_df["strand"] != "."]
    unstranded = peaks_df[peaks_df["strand"] == "."]
    if not stranded.empty:
        collect(_intersect(stranded, id_col, features, work_dir, "-s"))
    if not unstranded.empty:
        collect(_intersect(unstranded, id_col, features, work_dir, None))
    antisense: Set[str] = set()
    orphans = stranded[~stranded[id_col].isin(best)]
    if not orphans.empty:
        antisense = {rid for rid, *_ in _intersect(orphans, id_col, features, work_dir, "-S")}

    none = (0, "intergenic", ".", ".")
    out_df = peaks_df.copy()
    ids = out_df[id_col]
    out_df["region_type"] = [best.get(i, none)[1] for i in ids]
    out_df["gene_name"] = [best.get(i, none)[2] for i in ids]
    out_df["gene_type"] = [best.get(i, none)[3] for i in ids]
    out_df["antisense"] = [i in antisense for i in ids]

    if not quiet:
        log_breakdown(out_df)
    return out_df


def log_breakdown(df: pd.DataFrame, indent: str = "  ") -> None:
    counts = df["region_type"].value_counts()
    log.info(f"{indent}Region breakdown:")
    for rt, cnt in counts.items():
        log.info(f"{indent}  {rt:<15} : {cnt:>6,}  ({100*cnt/len(df):.1f}%)")
    if "antisense" in df and df["antisense"].any():
        n = int(df["antisense"].sum())
        log.info(f"{indent}  (of the intergenic: {n:,} lie antisense to a gene)")


def region_fractions(df: pd.DataFrame) -> Dict[str, float]:
    """Fraction of rows in each region class, plus 'antisense' (subset of intergenic)."""
    n = max(len(df), 1)
    out = {c: float((df["region_type"] == c).sum() / n) for c in REGION_CLASSES + ("intergenic",)}
    out["antisense"] = float(df["antisense"].sum() / n) if "antisense" in df else 0.0
    return out


def protein_coding_genes(gtf: Path, cache_dir: Path) -> Set[str]:
    """Names of protein-coding genes in the GTF (via the cached feature index)."""
    features = feature_bed_from_gtf(Path(gtf), cache_dir)
    genes: Set[str] = set()
    with open(features) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) > 6 and c[6] == "protein_coding" and c[3] != ".":
                genes.add(c[3])
    return genes


# ─────────────────────────────────────────────────────────────────────────────
# Stranded region sets for negatives
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd, stdin: Optional[str] = None) -> str:
    try:
        return subprocess.run(cmd, input=stdin, capture_output=True, text=True, check=True, env=_C_LOCALE).stdout
    except subprocess.CalledProcessError as e:
        raise PipelineError(f"{cmd[0]} {cmd[1]} failed while building region sets:\n{e.stderr}") from e
    except FileNotFoundError as e:
        raise PipelineError("bedtools not found. Install with: conda install -c bioconda bedtools") from e


def _merge_stranded(lines: Iterable[str], label: str) -> str:
    """Merge 6-column lines per strand; returns sorted 6-column BED text named `label`."""
    text = "".join(lines)
    if not text:
        return ""
    srt = _run(["sort", "-k1,1", "-k2,2n"], text)
    merged = _run(["bedtools", "merge", "-s", "-c", "6", "-o", "distinct", "-i", "stdin"], srt)
    return "".join(f"{c[0]}\t{c[1]}\t{c[2]}\t{label}\t0\t{c[3]}\n"
                   for c in (l.split("\t") for l in merged.splitlines()) if len(c) >= 4)


def _subtract(a: str, b: str, work_dir: Path) -> str:
    if not a or not b:
        return a
    fa, fb = work_dir / "_sub_a.bed", work_dir / "_sub_b.bed"
    fa.write_text(a); fb.write_text(b)
    try:
        return _run(["bedtools", "subtract", "-s", "-a", str(fa), "-b", str(fb)])
    finally:
        fa.unlink(missing_ok=True); fb.unlink(missing_ok=True)


def region_beds_from_gtf(gtf: Path, cache_dir: Path) -> Dict[str, Path]:
    """
    Disjoint, stranded region sets matching the annotation priorities:
      UTR      = UTR features
      exonic   = exon/CDS features minus UTR
      intronic = gene bodies minus exons
    Returns {class: 6-column BED path}, cached next to the feature index.
    """
    features = feature_bed_from_gtf(Path(gtf), cache_dir)
    out = {c: cache_dir / f"{Path(gtf).name}.regions.{FEATURE_INDEX_VERSION}.{c}.bed" for c in REGION_CLASSES}
    if all(p.exists() and p.stat().st_mtime >= features.stat().st_mtime for p in out.values()):
        return out

    log.info("  Building stranded UTR / exonic / intronic region sets from the GTF (one-off)...")
    by_class: Dict[str, list] = {"UTR": [], "exonic": [], "intronic": []}
    with open(features) as f:
        for line in f:
            c = line.split("\t")
            if c[5] not in "+-" or len(c[5]) != 1:
                continue
            row = f"{c[0]}\t{c[1]}\t{c[2]}\t.\t0\t{c[5]}\n"
            by_class[c[4]].append(row)
    utr = _merge_stranded(by_class["UTR"], "UTR")
    exons = _merge_stranded(by_class["exonic"] + by_class["UTR"], "exonic")
    genes = _merge_stranded(by_class["intronic"], "intronic")
    sets = {"UTR": utr, "exonic": _subtract(exons, utr, cache_dir), "intronic": _subtract(genes, exons, cache_dir)}
    for cls, text in sets.items():
        out[cls].write_text(text)
    return out


def bed_is_stranded(path: Path) -> bool:
    """True if every data line has a '+' or '-' in column 6."""
    seen = False
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 6 or c[5] not in ("+", "-"):
                return False
            seen = True
    return seen


def skip_annotation(peaks_df: pd.DataFrame, reason: str) -> pd.DataFrame:
    log.info(f"  Skipped ({reason})")
    out = peaks_df.copy()
    out["region_type"] = "not_annotated"
    out["gene_name"] = "."
    out["gene_type"] = "."
    out["antisense"] = False
    return out
