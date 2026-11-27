"""Run the whole analysis from a validated Config."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

from . import __version__
from .annotation import annotate_regions, skip_annotation
from .config import Config, validate
from .dataset import assign_splits, balance_splits
from .genome import Genome, check_assembly, check_peaks_within_genome, check_reference_naming
from .leakage import check_leakage
from .models import DEVICE, set_seed
from .negative_sampling import gc_summary, generate_candidates, select_gc_matched
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
    log.info(f"  Negatives    : {'provided ' + cfg.negatives.bed if cfg.negatives.bed else cfg.negatives.strategy}")
    log.info(f"  Test chroms  : {cfg.split.test_chroms}")
    log.info(f"  Val chroms   : {cfg.split.val_chroms}")
    log.info(f"  Device       : {DEVICE}")
    log.info("=" * 65)
    log.info(f"  Output dir   : {out_dir}")
    save_run_info(cfg, out_dir)

    genome = Genome(Path(cfg.reference.genome))
    regions_bed = None if cfg.negatives.bed else Path(cfg.reference.genes_bed)
    check_reference_naming(genome, regions_bed)
    check_assembly(genome, cfg.reference.assembly)

    # 1. Peaks
    peaks_df = validate_bed(Path(cfg.input), out_dir, cfg.peaks.min_log2fc, cfg.peaks.min_log10p, cfg.peaks.window)
    n_before = len(peaks_df)
    peaks_df = check_peaks_within_genome(peaks_df, genome)
    if len(peaks_df) != n_before:
        write_cleaned_bed(peaks_df, out_dir)

    # 2. Negatives (candidates)
    neg_bed = out_dir / f"{cfg.name}_negatives.bed"
    if cfg.negatives.bed:
        strategy = "provided"
        log.info(f"\n[2/9] Using provided negatives: {cfg.negatives.bed}")
        neg_df = read_bed(Path(cfg.negatives.bed), "neg")
    else:
        strategy = cfg.negatives.strategy
        rounds = cfg.negatives.gc_candidates if strategy == "gc-matched" else 1
        generate_candidates(out_dir / "peaks_cleaned.bed", genome, regions_bed, neg_bed, rounds, cfg.seed)
        neg_df = read_bed(neg_bed, "neg")

    # 3. Sequences (+ GC matching)
    pos_df, neg_df = extract_sequences(peaks_df, neg_df, genome, out_dir, cfg.model.max_len)
    if strategy == "gc-matched":
        log.info(f"  Selecting GC-matched negatives from {len(neg_df):,} candidates...")
        neg_df = select_gc_matched(pos_df, neg_df)
        neg_df[["chrom", "start", "end", "source_peak"]].assign(score=0, strand=neg_df["strand"]) \
            .to_csv(neg_bed, sep="\t", index=False, header=False)
        write_fasta(neg_df, "neg_id", out_dir / "negatives.fa")
        log.info(f"  GC-matched negatives: {len(neg_df):,}")
    pos_gc, neg_gc = gc_summary(pos_df, neg_df)

    # 4. Splits (+ leakage check) + models
    pos_df, neg_df = assign_splits(pos_df, neg_df, cfg.split.test_chroms, cfg.split.val_chroms)
    pos_df, neg_df, leakage, leaks = check_leakage(pos_df, neg_df, cfg.split.leakage, cfg.split.leakage_kmer)
    leaks.to_csv(out_dir / "leakage_flags.csv", index=False)
    data = balance_splits(pos_df, neg_df, cfg.seed)
    results = train_and_score(data, cfg, out_dir)
    results.update(negative_strategy=strategy, gc_positive=pos_gc, gc_negative=neg_gc,
                   n_test=len(data["test"][1]), leakage=leakage)

    # 5-8. Outputs
    make_plots(results, out_dir, cfg.name)
    log.info("\n[6/9] Annotating genomic regions...")
    if not cfg.steps.annotation:
        pos_df = skip_annotation(pos_df, "steps.annotation is false")
    elif not cfg.reference.gtf:
        pos_df = skip_annotation(pos_df, "no reference.gtf given")
    else:
        cache = Path(cfg.output_dir) / "_cache"
        pos_df = annotate_regions(pos_df, Path(cfg.reference.gtf), out_dir, cache)
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
