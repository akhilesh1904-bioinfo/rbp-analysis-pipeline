"""
Command line.

    rbp-pipeline run config.yaml [--set model.max_len=100 ...]
    rbp-pipeline default-config > my_protein.yaml
    rbp-pipeline validate config.yaml

`run_rbp_analysis.py` (the original flag-based interface) still works via `legacy_main`.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import PipelineError, __version__
from .config import Config, default_yaml, load_config, validate


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="rbp-pipeline", description="RBP binding-site analysis from eCLIP peaks")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the pipeline from a YAML config")
    run_p.add_argument("config", help="Path to YAML config")
    run_p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                       help="Override a config value, e.g. --set negatives.strategy=shuffle (repeatable)")

    val_p = sub.add_parser("validate", help="Check a config and its input files without running")
    val_p.add_argument("config")
    val_p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")

    sub.add_parser("default-config", help="Print a commented default config")

    args = p.parse_args(argv)
    try:
        if args.command == "default-config":
            sys.stdout.write(default_yaml())
            return 0
        cfg = load_config(args.config, args.overrides)
        if args.command == "validate":
            validate(cfg)
            print(f"OK: {args.config} is valid (results would go to {cfg.out_dir})")
            return 0
        from .pipeline import run   # heavy imports (torch) only when actually running
        run(cfg)
        return 0
    except PipelineError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible flag interface (run_rbp_analysis.py)
# ─────────────────────────────────────────────────────────────────────────────

def legacy_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generalised RBP analysis pipeline (flag interface; "
                                            "prefer `rbp-pipeline run config.yaml`)")
    p.add_argument("--input", required=True, help="Path to eCLIP BED file")
    p.add_argument("--name", required=True, help="Protein name (output folder/filenames)")
    p.add_argument("--genome", required=True, help="GRCh38 genome FASTA (chr-prefixed names)")
    p.add_argument("--gtf", default=None, help="(Optional) GENCODE GTF for region annotation")
    p.add_argument("--genes-bed", default=None, help="Regions negatives are drawn from (required unless --negatives)")
    p.add_argument("--output-dir", default="./results")
    p.add_argument("--negatives", default=None, help="Pre-computed negatives BED (skips generation)")
    p.add_argument("--negative-strategy", choices=["gc-matched", "shuffle"], default="gc-matched")
    p.add_argument("--gc-candidates", type=int, default=10)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--test-chroms", nargs="+", default=["chr21", "chr22", "chrX"])
    p.add_argument("--val-chroms", nargs="+", default=["chr19", "chr20"])
    p.add_argument("--min-log2fc", type=float, default=None)
    p.add_argument("--min-log10p", type=float, default=None)
    p.add_argument("--max-len", type=int, default=126)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-annotation", action="store_true")
    p.add_argument("--skip-report", action="store_true")
    return p


def config_from_legacy_args(args: argparse.Namespace) -> Config:
    cfg = Config(name=args.name, input=args.input, output_dir=args.output_dir, seed=args.seed)
    cfg.reference.genome, cfg.reference.genes_bed, cfg.reference.gtf = args.genome, args.genes_bed, args.gtf
    cfg.peaks.min_log2fc, cfg.peaks.min_log10p, cfg.peaks.window = args.min_log2fc, args.min_log10p, args.window
    cfg.negatives.strategy, cfg.negatives.gc_candidates = args.negative_strategy, args.gc_candidates
    cfg.negatives.bed = args.negatives
    cfg.split.test_chroms, cfg.split.val_chroms = list(args.test_chroms), list(args.val_chroms)
    cfg.model.max_len = args.max_len
    cfg.steps.annotation = not args.skip_annotation
    cfg.steps.report = not args.skip_report
    return cfg


def legacy_main(argv: Optional[List[str]] = None):
    """Parse the original flags, run, and return the results dict (exits 1 on PipelineError)."""
    cfg = config_from_legacy_args(legacy_parser().parse_args(argv))
    from .pipeline import run
    try:
        return run(cfg)
    except PipelineError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
