#!/usr/bin/env python3
"""
Backward-compatible entry point using the original command-line flags.

The preferred interface is now a YAML config:
    rbp-pipeline default-config > my_protein.yaml
    rbp-pipeline run my_protein.yaml

Example with flags:
    python run_rbp_analysis.py --input peaks.bed --name MY_PROTEIN \\
        --genome GRCh38.fa --genes-bed genes_only.bed [--gtf gencode.gtf]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))   # works without `pip install`

from rbp_pipeline.cli import legacy_main  # noqa: E402

if __name__ == "__main__":
    legacy_main()
