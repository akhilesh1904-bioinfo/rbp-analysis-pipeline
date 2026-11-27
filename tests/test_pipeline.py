"""End-to-end runs on synthetic genomes (need bedtools + samtools)."""
import json
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from conftest import MOTIF, ROOT, needs_tools, write_config, write_fasta
from rbp_pipeline.cli import legacy_main, main
from rbp_pipeline.config import load_config
from rbp_pipeline.genome import revcomp
from rbp_pipeline.pipeline import run


@needs_tools
@pytest.mark.parametrize("strategy", ["gc-matched", "shuffle"])
def test_full_pipeline_from_config(synthetic_config, strategy):
    cfg_path, d = synthetic_config
    results = run(load_config(cfg_path, [f"negatives.strategy={strategy}"]))
    out = d["tmp"] / "results" / "SYN"

    # The planted motif is learnable; the synthetic genome has uniform GC so GC alone is ~chance
    assert results["cnn_auc"] > 0.9 and results["lr_auc"] > 0.9
    assert 0.3 < results["gc_auc"] < 0.7
    assert results["negative_strategy"] == strategy

    csv = pd.read_csv(out / "SYN_predictions.csv")
    assert set(csv["split"]) == {"train", "val", "test"}
    assert set(csv.loc[csv.split == "test", "chrom"]) == {"chr21", "chr22", "chrX"}
    assert (csv["seen_in_training"] == (csv["split"] != "test")).all()

    # Each row's sequence is its own peak, in RNA sense
    genome = d["genome_seqs"]
    for r in csv.itertuples():
        seq = genome[r.chrom][r.start:r.end]
        assert r.sequence == (revcomp(seq) if r.strand == "-" else seq)
        assert MOTIF in r.sequence

    # Strand-aware annotation from the mini GTF (GENEA is + strand on chr21:0-20000)
    chr21 = csv[(csv.chrom == "chr21") & (csv.end <= 20000)]
    assert set(chr21.loc[chr21.strand == "+", "region_type"]) <= {"UTR", "exonic", "intronic"}
    assert set(chr21.loc[chr21.strand == "-", "region_type"]) == {"intergenic"}

    # Negatives are length- and chromosome-matched
    neg = pd.read_csv(out / "SYN_negatives.bed", sep="\t", header=None)
    pos = pd.read_csv(out / "peaks_cleaned.bed", sep="\t", header=None)
    assert neg[0].value_counts().sort_index().equals(pos[0].value_counts().sort_index())
    assert sorted((neg[2] - neg[1]).tolist()) == sorted((pos[2] - pos[1]).tolist())

    metrics = json.loads((out / "SYN_metrics.json").read_text())
    assert metrics["cnn_auc"] == pytest.approx(results["cnn_auc"])
    info = json.loads((out / "run_info.json").read_text())
    assert info["config"]["negatives"]["strategy"] == strategy and "torch" in info["versions"]
    assert "DONE" in (out / "pipeline.log").read_text()
    assert (out / "figures" / "cnn_training_curve.png").exists()

    # Interpretation: the CNN's most important k-mers and its top filters come from the planted motif
    motif_rna = MOTIF.replace("T", "U")
    s = results["interpretation"]
    assert s["cnn"]["status"] == "ok"
    assert all(k in motif_rna for k in s["cnn"]["top_kmers"][:3])
    assert any(f["consensus"].upper()[1:6] in motif_rna for f in s["cnn"]["top_filters"][:3])
    assert s["cnn"]["centre_vs_flank_importance"] > 3        # the motif sits at the peak centre
    assert s["streme"]["status"].startswith(("ok", "skipped"))
    idir = out / "interpretation"
    for f in ("cnn_kmer_importance.csv", "cnn_position_importance.csv", "cnn_filter_motifs.csv",
              "cnn_filter_motifs.meme", "cnn_importance.npz", "lr_kmer_coefficients.csv",
              "known_motif_enrichment.csv", "interpretation_summary.json"):
        assert (idir / f).exists(), f
    npz = __import__("numpy").load(idir / "cnn_importance.npz")
    assert set(npz["peak_id"]) <= set(csv.loc[csv.split == "test", "peak_id"])   # held-out peaks only
    for fig in ("interp_position_importance", "interp_top_sequences", "interp_cnn_filters", "interp_cnn_kmers",
                "interp_lr_kmers"):
        assert (out / "figures" / f"{fig}.png").exists(), fig


@needs_tools
def test_interpretation_can_be_switched_off(synthetic_config):
    cfg_path, d = synthetic_config
    res = run(load_config(cfg_path, ["steps.interpretation=false", "model.epochs=2", "steps.annotation=false"]))
    assert "interpretation" not in res
    assert not (d["tmp"] / "results" / "SYN" / "interpretation").exists()


@needs_tools
def test_legacy_flags_give_same_result_as_config(synthetic_config):
    cfg_path, d = synthetic_config
    via_config = run(load_config(cfg_path))
    via_flags = legacy_main([
        "--input", str(d["peaks_bed"]), "--name", "SYN_legacy", "--genome", str(d["genome"]),
        "--genes-bed", str(d["genes_bed"]), "--gtf", str(d["gtf"]), "--output-dir", str(d["tmp"] / "results"),
        "--min-log2fc", "3", "--min-log10p", "3", "--skip-report"])
    for k in ("cnn_auc", "lr_auc", "gc_auc"):
        assert via_flags[k] == pytest.approx(via_config[k])


@needs_tools
def test_cli_command_runs_and_reports_errors(synthetic_config, capsys):
    cfg_path, d = synthetic_config
    assert main(["run", str(cfg_path), "--set", "model.epochs=2", "--set", "steps.annotation=false"]) == 0
    assert main(["run", str(cfg_path), "--set", "split.test_chroms=[chr9]"]) == 1
    assert "test set is empty" in capsys.readouterr().err


@needs_tools
def test_legacy_script_still_runs(synthetic_dataset):
    d = synthetic_dataset
    r = subprocess.run([sys.executable, str(ROOT / "run_rbp_analysis.py"), "--input", str(d["peaks_bed"]),
                        "--name", "S", "--genome", str(d["genome"]), "--genes-bed", str(d["genes_bed"]),
                        "--output-dir", str(d["tmp"] / "r"), "--skip-report", "--skip-annotation"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "GC-only" in r.stdout


@needs_tools
def test_gc_matching_closes_a_gc_gap(tmp_path):
    """Peaks in GC-rich islands, AT-rich background: shuffle leaves a gap, gc-matched closes it."""
    rng = random.Random(1)
    chroms = ["chr1", "chr2", "chr19", "chr20", "chr21", "chr22", "chrX"]
    genome, peaks = {}, []
    for c in chroms:
        seq = [rng.choices("ACGT", weights=[3, 2, 2, 3])[0] for _ in range(60_000)]
        for i in range(30):
            start = 2_000 + i * 1_800
            for j in range(start - 200, start + 300):
                seq[j] = rng.choices("ACGT", weights=[15, 35, 35, 15])[0]
            peaks.append(f"{c}\t{start}\t{start + 80}\tp\t0\t{rng.choice('+-')}\t4\t5\n")
        for k in range(60):     # extra GC-rich islands with no peaks, so matched candidates exist
            s = 1_000 + k * 950 + 600
            for j in range(s, min(s + 150, 60_000)):
                seq[j] = rng.choices("ACGT", weights=[15, 35, 35, 15])[0]
        genome[c] = "".join(seq)
    write_fasta(tmp_path / "g.fa", genome)
    (tmp_path / "genes.bed").write_text("".join(f"{c}\t0\t60000\tg\t0\t+\n" for c in chroms))
    (tmp_path / "peaks.bed").write_text("".join(peaks))
    cfg_path = write_config(tmp_path / "c.yaml", {
        "name": "X", "input": "peaks.bed", "output_dir": str(tmp_path / "res"),
        "reference": {"genome": "g.fa", "genes_bed": "genes.bed"},
        "negatives": {"gc_candidates": 20}, "peaks": {"window": None}, "steps": {"report": False},
        "evaluation": {"bootstrap": 0}})

    gaps = {}
    for strategy in ("shuffle", "gc-matched"):
        res = run(load_config(cfg_path, [f"negatives.strategy={strategy}", f"name={strategy}"]))
        gaps[strategy] = (res["gc_positive"] - res["gc_negative"], res["gc_auc"])
    assert gaps["gc-matched"][0] < gaps["shuffle"][0] / 2
    assert gaps["gc-matched"][1] < gaps["shuffle"][1]


@needs_tools
def test_multi_seed_metrics_and_outputs(synthetic_config):
    cfg_path, d = synthetic_config
    res = run(load_config(cfg_path, ["evaluation.n_seeds=2", "model.epochs=3", "steps.annotation=false",
                                     "model.weight_decay=0.0001"]))
    out = d["tmp"] / "results" / "SYN"
    assert res["seeds"] == [42, 43]
    s = res["seed_summary"]["cnn"]["roc_auc"]
    assert len(s["values"]) == 2 and s["sd"] >= 0
    m = res["metrics"]
    assert set(m) == {"cnn", "lr", "ensemble", "gc_baseline"}
    assert {"pr_auc", "f1", "mcc", "brier", "confusion", "roc_auc_ci"} <= set(m["cnn"])
    assert "f1" not in m["gc_baseline"]                       # GC fraction is not a probability
    lo, hi = m["cnn"]["roc_auc_ci"]
    assert lo <= m["cnn"]["roc_auc"] <= hi
    assert set(res["comparisons"]) == {"cnn_vs_lr", "cnn_vs_gc_baseline", "lr_vs_gc_baseline"}

    saved = json.loads((out / "SYN_metrics.json").read_text())
    assert saved["seed_summary"]["cnn"]["roc_auc"]["values"] == s["values"]
    table = pd.read_csv(out / "SYN_metrics.csv")
    assert {"model", "metric", "value", "ci_low", "ci_high", "seed_mean", "seed_sd"} <= set(table.columns)
    for fig in ("pr_curve", "calibration", "confusion_matrices", "roc_curve"):
        assert (out / "figures" / f"{fig}.png").exists(), fig
    assert "leakage" in saved and (out / "leakage_flags.csv").exists()


@needs_tools
def test_leakage_remove_mode_runs(synthetic_config):
    """Copy some training peaks onto test chromosomes; remove mode must drop them from the test set."""
    cfg_path, d = synthetic_config
    genome = d["genome_seqs"]
    # duplicate 10 chr1 peak sequences into chr21 at fresh positions (forward strand)
    peaks = pd.read_csv(d["peaks_bed"], sep="\t", header=None)
    chr21 = list(genome["chr21"])
    extra = []
    for i, r in enumerate(peaks[peaks[0] == "chr1"].head(10).itertuples()):
        seq = genome["chr1"][r[2]:r[3]]
        seq = revcomp(seq) if r[6] == "-" else seq
        pos = 37_000 + i * 250
        chr21[pos:pos + len(seq)] = list(seq)
        extra.append(f"chr21\t{pos}\t{pos + len(seq)}\tdup{i}\t1000\t+\t4\t5\n")
    genome["chr21"] = "".join(chr21)
    write_fasta(d["genome"], genome)
    Path(str(d["genome"]) + ".fai").unlink(missing_ok=True)
    with open(d["peaks_bed"], "a") as f:
        f.writelines(extra)

    res = run(load_config(cfg_path, ["split.leakage=remove", "steps.annotation=false", "model.epochs=2"]))
    assert res["leakage"]["test"]["by_reason"].get("exact", 0) >= 10
    flags = pd.read_csv(d["tmp"] / "results" / "SYN" / "leakage_flags.csv")
    preds = pd.read_csv(d["tmp"] / "results" / "SYN" / "SYN_predictions.csv")
    test_ids = set(preds.loc[preds.split == "test", "peak_id"])
    assert test_ids.isdisjoint(set(flags["seq_id"]))
