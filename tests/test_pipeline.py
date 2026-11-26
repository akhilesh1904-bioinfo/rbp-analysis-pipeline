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
@pytest.mark.parametrize("strategy", ["gc-matched", "shuffle"])
def test_full_pipeline(synthetic_dataset, strategy):
    d = synthetic_dataset
    out_root = d["tmp"] / "results"
    results = rbp.main([
        "--input", str(d["peaks_bed"]), "--name", "SYN",
        "--genome", str(d["genome"]), "--genes-bed", str(d["genes_bed"]),
        "--output-dir", str(out_root), "--skip-report", "--min-log2fc", "3", "--min-log10p", "3",
        "--negative-strategy", strategy,
    ])
    out = out_root / "SYN"

    # The planted motif is learnable, so a working pipeline separates it well
    assert results["lr_auc"] > 0.9
    assert results["cnn_auc"] > 0.9
    # The synthetic genome has uniform GC, so GC alone should be near chance
    assert 0.3 < results["gc_auc"] < 0.7
    assert results["negative_strategy"] == strategy

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
    # Negatives are length-matched to their source peak
    assert sorted((neg[2] - neg[1]).tolist()) == sorted((pos[2] - pos[1]).tolist())

    info = json.loads((out / "run_info.json").read_text())
    assert info["seed"] == 42 and "torch" in info["versions"]


@needs_tools
def test_gc_matching_closes_a_gc_gap(tmp_path):
    """Peaks in a GC-rich block, background AT-rich: shuffle leaves a gap, gc-matched closes it."""
    import random
    from conftest import write_fasta
    rng = random.Random(1)
    chroms = ["chr1", "chr2", "chr19", "chr20", "chr21", "chr22", "chrX"]
    genome, peaks = {}, []
    for c in chroms:
        # 40% GC background with a few 70%-GC islands where the peaks sit
        seq = [rng.choices("ACGT", weights=[3, 2, 2, 3])[0] for _ in range(60_000)]
        for i in range(30):
            start = 2_000 + i * 1_800
            for j in range(start - 200, start + 300):
                seq[j] = rng.choices("ACGT", weights=[15, 35, 35, 15])[0]
            peaks.append(f"{c}\t{start}\t{start + 80}\tp\t0\t{rng.choice('+-')}\t4\t5\n")
        # extra GC-rich islands with no peaks, so GC-matched candidates exist
        for k in range(60):
            s = 1_000 + k * 950 + 600
            for j in range(s, min(s + 150, 60_000)):
                seq[j] = rng.choices("ACGT", weights=[15, 35, 35, 15])[0]
        genome[c] = "".join(seq)
    fa = write_fasta(tmp_path / "g.fa", genome)
    genes = tmp_path / "genes.bed"
    genes.write_text("".join(f"{c}\t0\t60000\tg\t0\t+\n" for c in chroms))
    bed = tmp_path / "peaks.bed"
    bed.write_text("".join(peaks))

    gaps = {}
    for strategy in ("shuffle", "gc-matched"):
        args = ["--input", str(bed), "--name", strategy, "--genome", str(fa), "--genes-bed", str(genes),
                "--output-dir", str(tmp_path / "res"), "--skip-report", "--negative-strategy", strategy,
                "--gc-candidates", "20"]
        res = rbp.main(args)
        pos_gc, neg_gc = (float(x.rstrip("%")) for x in res["gc_summary"].split(" / "))
        gaps[strategy] = (pos_gc - neg_gc, res["gc_auc"])
    assert gaps["gc-matched"][0] < gaps["shuffle"][0] / 2
    assert gaps["gc-matched"][1] < gaps["shuffle"][1]
