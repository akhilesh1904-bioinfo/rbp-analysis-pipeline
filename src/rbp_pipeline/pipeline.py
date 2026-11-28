"""Run the whole analysis from a validated Config."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from . import __version__
from .annotation import annotate_regions, log_breakdown, region_beds_from_gtf, skip_annotation
from .config import Config, validate
from .dataset import assign_splits, balance_splits_grouped
from .genome import Genome, check_assembly, check_peaks_within_genome, check_reference_naming
from .leakage import check_leakage
from .models import DEVICE, set_seed
from .negative_sampling import (gc_summary, generate_candidates, log_matching, matching_report, region_qc,
                                select_gc_matched)
from .preprocessing import read_bed, validate_bed, write_cleaned_bed
from .reporting import (generate_word_report, make_plots, save_metrics, save_predictions,
                        save_run_info)
from .sequence_extraction import extract_sequences, write_fasta
from .training import train_and_score

log = logging.getLogger("rbp_pipeline")


def setup_logging(out_dir: Path) -> None:
    """Log plain messages to the console and to <out_dir>/pipeline.log."""
    log.handlers.clear()
    log.setLevel(logging.INFO)
    log.propagate = False
    fmt = logging.Formatter("%(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / "pipeline.log", mode="w")):
        handler.setFormatter(fmt)
        log.addHandler(handler)


def run(cfg: Config) -> Dict[str, Any]:
    validate(cfg)
    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    setup_logging(out_dir)
    set_seed(cfg.seed)

    log.info("=" * 65)
    log.info(f"  RBP ANALYSIS PIPELINE v{__version__}")
    log.info(f"  Protein      : {cfg.name}")
    log.info(f"  Input        : {cfg.input}")
    log.info(f"  Genome       : {cfg.reference.genome}")
    n = cfg.negatives
    neg_desc = (f"provided {n.bed}" if n.bed else
                f"{n.strategy}, from {'GTF ' + '/'.join(n.region_classes) if n.source == 'gtf' else 'genes_bed'}"
                + (", same strand" if n.strand_aware else "") + (", same region class" if n.match_region else ""))
    log.info(f"  Negatives    : {neg_desc}")
    log.info(f"  Test chroms  : {cfg.split.test_chroms}")
    log.info(f"  Val chroms   : {cfg.split.val_chroms}")
    log.info(f"  Device       : {DEVICE}")
    log.info("=" * 65)
    log.info(f"  Output dir   : {out_dir}")
    save_run_info(cfg, out_dir)

    genome = Genome(Path(cfg.reference.genome))
    regions_bed = None if (n.bed or n.source == "gtf") else Path(cfg.reference.genes_bed)
    check_reference_naming(genome, regions_bed)
    gtf = Path(cfg.reference.gtf) if cfg.reference.gtf else None
    cache = Path(cfg.output_dir) / "_cache"
    check_assembly(genome, cfg.reference.assembly)

    # 1. Peaks
    peaks_df = validate_bed(Path(cfg.input), out_dir, cfg.peaks.min_log2fc, cfg.peaks.min_log10p, cfg.peaks.window)
    n_before = len(peaks_df)
    peaks_df = check_peaks_within_genome(peaks_df, genome)
    if len(peaks_df) != n_before:
        write_cleaned_bed(peaks_df, out_dir)

    # Region annotation of positives (needed early for region-matched negatives and the region QC)
    pos_annot = None
    if gtf and (cfg.steps.annotation or (n.qc and not n.bed) or n.match_region):
        pos_annot = annotate_regions(peaks_df, gtf, out_dir, cache, quiet=True)[
            ["peak_id", "region_type", "gene_name", "gene_type", "antisense"]]

    # 2. Negatives (candidates)
    neg_bed = out_dir / f"{cfg.name}_negatives.bed"
    if cfg.negatives.bed:
        strategy = "provided"
        log.info(f"\n[2/9] Using provided negatives: {cfg.negatives.bed}")
        neg_df = read_bed(Path(cfg.negatives.bed), "neg")
    else:
        strategy = n.strategy
        rounds = n.gc_candidates if strategy == "gc-matched" else 1
        groups = None
        if n.source == "gtf":
            beds = region_beds_from_gtf(gtf, cache)
            regions = {c: beds[c] for c in n.region_classes}
            if n.match_region:
                groups = dict(zip(pos_annot["peak_id"], pos_annot["region_type"]))
        else:
            regions = regions_bed
        generate_candidates(out_dir / "peaks_cleaned.bed", genome, regions, neg_bed, rounds, cfg.seed,
                            strand_aware=n.strand_aware, peak_groups=groups)
        neg_df = read_bed(neg_bed, "neg")

    # 3. Sequences (+ GC matching)
    pos_df, neg_df = extract_sequences(peaks_df, neg_df, genome, out_dir, cfg.model.max_len)
    if strategy == "gc-matched":
        log.info(f"  Selecting GC-matched negatives from {len(neg_df):,} candidates...")
        neg_df = select_gc_matched(pos_df, neg_df, n.max_gc_diff)
        neg_df[["chrom", "start", "end", "source_peak"]].assign(score=0, strand=neg_df["strand"]) \
            .to_csv(neg_bed, sep="\t", index=False, header=False)
        write_fasta(neg_df, "neg_id", out_dir / "negatives.fa")
        log.info(f"  GC-matched negatives: {len(neg_df):,}")
    pos_gc, neg_gc = gc_summary(pos_df, neg_df)
    matching, unmatched = matching_report(pos_df, neg_df, pos_annot)
    log_matching(matching)
    if "n_unmatched" in matching:
        unmatched.to_csv(out_dir / "unmatched_peaks.csv", index=False)

    # Region / strand QC of the negatives
    negatives_qc = None
    if n.qc and gtf and pos_annot is not None:
        neg_df = annotate_regions(neg_df, gtf, out_dir, cache, id_col="neg_id", quiet=True)
        negatives_qc = region_qc(pos_annot[pos_annot["peak_id"].isin(pos_df["peak_id"])], neg_df)
        _save_qc(negatives_qc, out_dir / "negatives_qc.csv")
    elif n.qc and not gtf:
        log.info("  Region QC skipped (needs reference.gtf)")

    # 4. Splits (+ leakage check) + models
    pos_df, neg_df = assign_splits(pos_df, neg_df, cfg.split.test_chroms, cfg.split.val_chroms)
    pos_df, neg_df, leakage, leaks = check_leakage(pos_df, neg_df, cfg.split.leakage, cfg.split.leakage_kmer)
    leaks.to_csv(out_dir / "leakage_flags.csv", index=False)
    gene_of = dict(zip(pos_annot["peak_id"], pos_annot["gene_name"])) if pos_annot is not None else None
    data, groups, balance = balance_splits_grouped(pos_df, neg_df, cfg.seed, cfg.split.balance, gene_of)
    test_groups = _bootstrap_groups(cfg, groups["test"], balance["mode"], gene_of is not None)
    results = train_and_score(data, cfg, out_dir, test_groups)
    results.update(negative_strategy=strategy, gc_positive=pos_gc, gc_negative=neg_gc,
                   n_test=len(data["test"][1]), leakage=leakage, negatives_qc=negatives_qc,
                   matching=matching, balance=balance)

    # 5-8. Outputs
    make_plots(results, out_dir, cfg.name)
    log.info("\n[6/9] Annotating genomic regions...")
    if not cfg.steps.annotation:
        pos_df = skip_annotation(pos_df, "steps.annotation is false")
    elif pos_annot is None:
        pos_df = skip_annotation(pos_df, "no reference.gtf given")
    else:
        pos_df = pos_df.merge(pos_annot, on="peak_id", how="left")
        log_breakdown(pos_df)
    pos_df = save_predictions(pos_df, results, cfg, out_dir)
    if cfg.steps.interpretation:
        from .interpretation import run_interpretation
        results["interpretation"] = run_interpretation(pos_df, neg_df, data, results, cfg, out_dir)
    else:
        log.info("\n[8/9] Interpretation skipped (steps.interpretation is false)")
    save_metrics(results, out_dir, cfg.name)

    if cfg.steps.report:
        generate_word_report(pos_df, results, cfg, out_dir)
    else:
        log.info("\n[9/9] Word report skipped (steps.report is false)")

    log.info("\n" + "=" * 65)
    log.info("  DONE")
    log.info("=" * 65)
    log.info(f"  CNN  AUC : {results['cnn_auc']:.4f}")
    log.info(f"  LR   AUC : {results['lr_auc']:.4f}")
    log.info(f"  Mean AUC : {results['mean_auc']:.4f}")
    log.info(f"  GC-only  : {results['gc_auc']:.4f}  (baseline)")
    log.info(f"\n  Results  : {out_dir}")
    log.info("=" * 65)
    return results


def _save_qc(qc: Dict[str, Any], path: Path) -> None:
    import pandas as pd
    rows = [{"region": k, "positives": qc["positives"][k], "negatives": qc["negatives"][k]} for k in qc["positives"]]
    pd.DataFrame(rows).to_csv(path, index=False)


def _bootstrap_groups(cfg: Config, test: Dict[str, List[str]], balance_mode: str, have_genes: bool):
    """Bootstrap unit label per test sequence, or None to resample sequences individually."""
    unit = cfg.evaluation.bootstrap_unit
    if unit == "sequence":
        return None
    if balance_mode != "paired":
        log.info(f"  Note: evaluation.bootstrap_unit: {unit} needs paired balancing; resampling sequences instead")
        return None
    if unit == "gene" and not have_genes:
        log.info("  Note: evaluation.bootstrap_unit: gene needs reference.gtf; resampling pairs instead")
        return test["pair"]
    return test[unit]
