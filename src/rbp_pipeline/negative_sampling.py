"""Negative (non-binding) region generation: length-, chromosome-, strand- and optionally GC- and region-matched."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from bisect import bisect_right
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from . import PipelineError
from .annotation import bed_is_stranded
from .genome import GRCH38_SIZES, VALID_CHROMS, Genome, gc_fraction

log = logging.getLogger("rbp_pipeline")


def _data_lines(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith(("#", "track", "browser")):
                yield line


def generate_candidates(peaks_bed: Path, genome: Genome, regions: Union[Path, Dict[str, Path]],
                        out_bed: Path, rounds: int, seed: int, strand_aware: bool = True,
                        peak_groups: Optional[Dict[str, str]] = None) -> Dict[str, int]:
    """
    Place every peak `rounds` times inside candidate regions on its own chromosome (each placement
    lies entirely within a region); column 4 keeps the source peak_id.

    strand_aware: a '+' peak is only placed in '+' regions and a '-' peak in '-' regions, so its
        sequence (extracted on the peak's strand) is sense transcript, never the reverse complement.
        Needs regions with a strand in column 6; without one a warning is logged.
    regions: one BED, or {region class: BED}. With a dict and `peak_groups` (peak_id -> class),
        each peak is shuffled into regions of its own class (falling back to all classes when its
        class has no regions on that chromosome and strand).

    bedtools' own -chrom flag is silently ignored when combined with -incl (v2.31), so we
    shuffle one chromosome at a time instead.
    """
    label = "candidate" if rounds > 1 else "matched negative"
    log.info(f"\n[2/9] Generating {label} regions ({rounds} per peak)...")
    work = out_bed.parent

    chrom_sizes = {c: s for c, s in genome.chrom_sizes().items() if c in VALID_CHROMS}
    if not chrom_sizes:
        log.info("  Note: genome .fai unavailable — using standard GRCh38 chromosome sizes")
        chrom_sizes = GRCH38_SIZES

    region_files = {"all": regions} if isinstance(regions, (str, Path)) else dict(regions)
    stranded = strand_aware and all(bed_is_stranded(Path(p)) for p in region_files.values())
    if strand_aware and not stranded:
        log.warning("  WARNING: the region BED has no strand column (+/- in column 6), so negatives can land on "
                    "the opposite strand of a gene and be extracted as antisense sequence. Use a stranded BED "
                    "(rbp-pipeline make-regions) or negatives.source: gtf.")

    # regions[(class, chrom, strand)] -> lines ; class 'all' holds the union
    region_lines: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for cls, path in region_files.items():
        for line in _data_lines(Path(path)):
            c = line.rstrip("\n").split("\t")
            strand = c[5] if stranded else "*"
            region_lines[(cls, c[0], strand)].append(line)
            if cls != "all":
                region_lines[("all", c[0], strand)].append(line)

    jobs: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    n_fallback = 0
    for line in _data_lines(peaks_bed):
        c = line.rstrip("\n").split("\t")
        chrom, pid = c[0], c[3]
        pstrand = c[5] if len(c) > 5 else "."
        skey = pstrand if stranded and pstrand in ("+", "-") else "*"
        group = (peak_groups or {}).get(pid, "all")
        if group != "all" and not _has_regions(region_lines, group, chrom, skey):
            group = "all"
            n_fallback += 1
        jobs[(chrom, group, skey)].append(line)

    chrom_index = {c: i for i, c in enumerate(sorted({k[0] for k in jobs}))}
    genome_file, chrom_peaks, chrom_regions = (work / "_genome.txt", work / "_shuffle_peaks.bed",
                                               work / "_shuffle_regions.bed")
    outputs, skipped, n_no_regions, n_short = [], set(), 0, 0
    sub_index: Dict[str, int] = defaultdict(int)
    try:
        for (chrom, group, skey) in sorted(jobs):
            peaks = jobs[(chrom, group, skey)]
            sub = sub_index[chrom]
            sub_index[chrom] += 1
            regs = _regions_for(region_lines, group, chrom, skey)
            if chrom not in chrom_sizes or not regs:
                skipped.add(chrom)
                n_no_regions += len(peaks)
                continue
            genome_file.write_text(f"{chrom}\t{chrom_sizes[chrom]}\n")
            chrom_regions.write_text("".join(regs))
            kept, short = _shuffle_contained(peaks, regs, rounds, genome_file, chrom_peaks, chrom_regions,
                                             peaks_bed, seed + chrom_index[chrom] + 100_000 * sub, chrom)
            outputs.append("".join(kept))
            n_short += short
    finally:
        for p in (genome_file, chrom_peaks, chrom_regions):
            p.unlink(missing_ok=True)

    if n_short:
        log.info(f"  Note: {n_short:,} peaks got fewer than {rounds} candidates fully inside a region")
    if skipped:
        log.warning(f"  WARNING: {n_no_regions:,} peaks had no candidate regions (chromosome/strand"
                    f"{'/class' if peak_groups else ''}) on {sorted(skipped)}; no negatives generated for them")
    out_bed.write_text("".join(outputs))
    n = sum(len([l for l in o.splitlines() if l.strip()]) for o in outputs)
    how = ["length- and chromosome-matched"]
    if stranded:
        how.append("same strand")
    if peak_groups:
        how.append("same region class")
    log.info(f"  Regions generated   : {n:,} ({', '.join(how)})")
    if n_fallback:
        log.info(f"  Note: {n_fallback:,} peaks (e.g. intergenic) drawn from all region classes")
    return {"n_candidates": n, "stranded": int(stranded), "n_fallback": n_fallback, "n_no_regions": n_no_regions}


def _merged_intervals(lines: List[str]) -> Tuple[List[int], List[int]]:
    iv = sorted((int(c[1]), int(c[2])) for c in (l.split("\t", 3) for l in lines))
    starts, ends = [], []
    for a, b in iv:
        if starts and a <= ends[-1]:
            ends[-1] = max(ends[-1], b)
        else:
            starts.append(a); ends.append(b)
    return starts, ends


def _contained(starts: List[int], ends: List[int], a: int, b: int) -> bool:
    i = bisect_right(starts, a) - 1
    return i >= 0 and b <= ends[i]


def _shuffle_contained(peaks: List[str], regs: List[str], rounds: int, genome_file: Path, chrom_peaks: Path,
                       chrom_regions: Path, excl: Path, seed: int, chrom: str,
                       oversample: int = 2, max_attempts: int = 8) -> Tuple[List[str], int]:
    """
    `rounds` placements per peak that lie entirely inside a region. bedtools shuffle -incl only
    guarantees the start is inside, so placements are oversampled and filtered; peaks still short
    after `max_attempts` keep what they have.
    """
    starts, ends = _merged_intervals(regs)
    need = {l.split("\t", 4)[3]: rounds for l in peaks}
    line_of = {l.split("\t", 4)[3]: l for l in peaks}
    kept: List[str] = []
    for attempt in range(max_attempts):
        pending = [pid for pid, n in need.items() if n > 0]
        if not pending:
            break
        chrom_peaks.write_text("".join(line_of[pid] * (need[pid] * oversample) for pid in pending))
        cmd = ["bedtools", "shuffle", "-i", str(chrom_peaks), "-g", str(genome_file), "-incl", str(chrom_regions),
               "-excl", str(excl), "-maxTries", "1000", "-seed", str(seed + 1000 * attempt)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise PipelineError(f"bedtools shuffle failed on {chrom}:\n{e.stderr}") from e
        except FileNotFoundError as e:
            raise PipelineError("bedtools not found. Install with: conda install -c bioconda bedtools") from e
        for line in result.stdout.splitlines(keepends=True):
            c = line.split("\t", 4)
            if len(c) < 4:
                continue
            pid = c[3]
            if need.get(pid, 0) > 0 and _contained(starts, ends, int(c[1]), int(c[2])):
                kept.append(line)
                need[pid] -= 1
    return kept, sum(1 for n in need.values() if n > 0)


def _regions_for(region_lines, group, chrom, skey) -> List[str]:
    if skey == "*":
        return [l for (g, c, _s), lines in region_lines.items() if g == group and c == chrom for l in lines]
    return region_lines.get((group, chrom, skey), [])


def _has_regions(region_lines, group, chrom, skey) -> bool:
    return bool(_regions_for(region_lines, group, chrom, skey))


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


def region_qc(pos_annot: pd.DataFrame, neg_annot: pd.DataFrame, max_distance: float = 0.15,
              max_antisense: float = 0.05) -> Dict:
    """
    Compare where positives and negatives lie (UTR / exonic / intronic / intergenic, plus the share of
    negatives antisense to a gene). Warns when the negatives would let a model separate the classes by
    region or strand instead of by binding. `distance` is the total variation distance (0 = identical,
    1 = disjoint) between the two region distributions.
    """
    from .annotation import REGION_CLASSES, region_fractions
    pos, neg = region_fractions(pos_annot), region_fractions(neg_annot)
    classes = list(REGION_CLASSES) + ["intergenic"]
    distance = 0.5 * sum(abs(pos[c] - neg[c]) for c in classes)
    log.info("  Region QC (positives vs negatives):")
    for c in classes:
        log.info(f"    {c:<12}: {100*pos[c]:5.1f}% vs {100*neg[c]:5.1f}%")
    log.info(f"    antisense   : {100*pos['antisense']:5.1f}% vs {100*neg['antisense']:5.1f}%   "
             f"(distance {distance:.2f})")
    class_match = None
    if "source_peak" in neg_annot and "peak_id" in pos_annot:
        peak_class = dict(zip(pos_annot["peak_id"], pos_annot["region_type"]))
        src = neg_annot["source_peak"].map(peak_class)
        genic = src.isin(REGION_CLASSES)
        if genic.any():
            class_match = float((neg_annot.loc[genic, "region_type"] == src[genic]).mean())
            log.info(f"    negatives in the same region class as their peak: {100*class_match:.1f}% "
                     "(genic peaks)")
    warnings = []
    if neg["antisense"] > max_antisense:
        warnings.append(f"{100*neg['antisense']:.1f}% of negatives lie antisense to a gene: their sequence is the "
                        "reverse complement of a transcript. Use a stranded region BED or negatives.source: gtf.")
    if distance > max_distance:
        hint = ("Consider negatives.source: gtf with negatives.match_region: true."
                if class_match is None or class_match < 0.9 else
                f"{100*pos['intergenic']:.0f}% of positives are intergenic and can't be matched by genic regions.")
        warnings.append(f"positives and negatives differ in region class (distance {distance:.2f} > {max_distance}); "
                        f"a model can partly separate them by region. {hint}")
    for w in warnings:
        log.warning(f"  WARNING: {w}")
    return {"positives": pos, "negatives": neg, "distance": distance, "class_match": class_match,
            "warnings": warnings}
