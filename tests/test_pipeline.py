"""End-to-end run on a synthetic genome (needs bedtools + samtools)."""
import json
import shutil

import pandas as pd
import pytest

import run_rbp_analysis as rbp
from conftest import MOTIF

needs_tools = pytest.mark.skipif(
    shutil.which("bedtools") is None or shutil.which("samtools") is None,
    reason="bedtools and samtools required")


@needs_tools
def test_full_pipeline(synthetic_dataset):
    d = synthetic_dataset
    out_root = d["tmp"] / "results"
    results = rbp.main([
        "--input", str(d["peaks_bed"]), "--name", "SYN",
        "--genome", str(d["genome"]), "--genes-bed", str(d["genes_bed"]),
        "--output-dir", str(out_root), "--skip-report", "--min-log2fc", "3", "--min-log10p", "3",
    ])
    out = out_root / "SYN"

    # The planted motif is learnable, so a working pipeline separates it well
    assert results["lr_auc"] > 0.9
    assert results["cnn_auc"] > 0.9

    csv = pd.read_csv(out / "SYN_predictions.csv")
    # Bug 3: every row says which split it came from; only test-chrom peaks are unseen
    assert set(csv["split"]) == {"train", "val", "test"}
    assert set(csv.loc[csv.split == "test", "chrom"]) == {"chr21", "chr22", "chrX"}
    assert (csv["seen_in_training"] == (csv["split"] != "test")).all()

    # Bugs 2 + 4: each row's sequence is its own peak, in RNA sense
    genome = d["genome_seqs"]
    for r in csv.itertuples():
        seq = genome[r.chrom][r.start:r.end]
        if r.strand == "-":
            seq = rbp.revcomp(seq)
        assert r.sequence == seq
        assert MOTIF in r.sequence

    # Negatives are chromosome-matched (bedtools shuffle -chrom)
    neg = pd.read_csv(out / "SYN_negatives.bed", sep="\t", header=None)
    pos = pd.read_csv(out / "peaks_cleaned.bed", sep="\t", header=None)
    assert neg[0].value_counts().sort_index().equals(pos[0].value_counts().sort_index())

    info = json.loads((out / "run_info.json").read_text())
    assert info["seed"] == 42 and "torch" in info["versions"]
