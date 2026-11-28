"""Chromosome splits, encodings, and reproducible training."""
import numpy as np
import pandas as pd
import pytest
import torch

from rbp_pipeline import PipelineError
from rbp_pipeline.config import ModelConfig
from rbp_pipeline.dataset import build_splits, fit_length, one_hot, seq_to_kmers
from rbp_pipeline.models import train_cnn


def make_df(chroms, prefix):
    return pd.DataFrame({"chrom": chroms, "sequence": [f"{prefix}{i}" for i in range(len(chroms))]})


def test_splits_are_chromosome_disjoint_and_balanced():
    chroms = ["chr1"] * 10 + ["chr19"] * 4 + ["chr21"] * 3
    data, pos_out, _ = build_splits(make_df(chroms, "P"), make_df(chroms + ["chr21"] * 5, "N"),
                                    ["chr21"], ["chr19"], seed=0)
    assert set(pos_out.loc[pos_out.split == "test", "chrom"]) == {"chr21"}
    assert set(pos_out.loc[pos_out.split == "val", "chrom"]) == {"chr19"}
    for seqs, labels in data.values():
        assert sum(labels) * 2 == len(labels)
    assert set(data["test"][0]).isdisjoint(data["train"][0])


def test_overlapping_test_and_val_rejected():
    pos = make_df(["chr1", "chr21"], "P")
    with pytest.raises(PipelineError, match="both test and val"):
        build_splits(pos, pos, ["chr21"], ["chr21"], seed=0)


def test_empty_val_rejected():
    pos = make_df(["chr1", "chr21"], "P")
    with pytest.raises(PipelineError, match="val set is empty"):
        build_splits(pos, pos, ["chr21"], ["chr19"], seed=0)


def test_centre_crop():
    assert fit_length("AAACCCGGG", 3) == "CCC"
    assert fit_length("ACG", 10) == "ACG"


def test_one_hot_uses_centre_and_ignores_n():
    arr = one_hot("TTACGNTT", 4)      # centre 4 bp = "ACGN"
    assert arr[0, 0] == 1 and arr[1, 1] == 1 and arr[2, 2] == 1 and arr[:, 3].sum() == 0


def test_kmers_skip_n():
    assert "N" not in seq_to_kmers("ACNGT")


def test_cnn_training_is_reproducible():
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), 40)) for _ in range(64)]
    labels = [i % 2 for i in range(64)]
    cfg = ModelConfig(max_len=40, epochs=2)
    runs = [train_cnn((seqs, labels), (seqs[:16], labels[:16]), cfg, seed=3)[0].state_dict() for _ in range(2)]
    for k in runs[0]:
        assert torch.equal(runs[0][k], runs[1][k]), k


def test_weight_decay_and_dropout_change_training():
    rng = np.random.default_rng(1)
    seqs = ["".join(rng.choice(list("ACGT"), 40)) for _ in range(64)]
    labels = [i % 2 for i in range(64)]
    base = train_cnn((seqs, labels), (seqs[:16], labels[:16]), ModelConfig(max_len=40, epochs=2), seed=3)[0]
    reg = train_cnn((seqs, labels), (seqs[:16], labels[:16]),
                    ModelConfig(max_len=40, epochs=2, weight_decay=0.1, dropout=0.5), seed=3)[0]
    assert reg.classifier[2].p == 0.5
    assert not torch.equal(base.conv1.weight, reg.conv1.weight)


def _paired_frames():
    # 6 peaks on chr1 (train), 2 on chr19 (val), 3 on chr21 (test); p10 has no negative,
    # and n_orphan's peak is not in the data (e.g. removed by the leakage check).
    chroms = ["chr1"] * 6 + ["chr19"] * 2 + ["chr21"] * 3
    pos = pd.DataFrame({"peak_id": [f"p{i}" for i in range(11)], "chrom": chroms,
                        "sequence": [f"P{i}" for i in range(11)], "split": None})
    neg = pd.DataFrame({"neg_id": [f"n{i}" for i in range(10)] + ["n_orphan"],
                        "source_peak": [f"p{i}" for i in range(10)] + ["p_gone"],
                        "chrom": chroms[:10] + ["chr21"], "sequence": [f"N{i}" for i in range(10)] + ["NX"]})
    from rbp_pipeline.dataset import assign_splits
    return assign_splits(pos, neg, ["chr21"], ["chr19"])


def test_paired_balancing_keeps_matched_pairs_together():
    from rbp_pipeline.dataset import balance_splits_grouped
    pos, neg = _paired_frames()
    genes = {"p8": "GENEA", "p9": "GENEA", "p10": "GENEB"}
    data, groups, summary = balance_splits_grouped(pos, neg, seed=0, mode="paired", gene_of=genes)
    seqs, labels = data["test"]
    assert sorted(seqs) == ["N8", "N9", "P8", "P9"]          # p10 (no negative) and the orphan dropped
    pair = dict(zip(seqs, groups["test"]["pair"]))
    assert pair["P8"] == pair["N8"] == "p8" and pair["P9"] == pair["N9"] == "p9"
    assert set(groups["test"]["gene"]) == {"gene:GENEA"}
    assert summary["test"]["positives_dropped"] == 1 and summary["test"]["negatives_dropped"] == 1
    assert summary["mode"] == "paired"


def test_paired_balancing_falls_back_without_source_peaks():
    from rbp_pipeline.dataset import balance_splits_grouped
    pos, neg = _paired_frames()
    data, groups, summary = balance_splits_grouped(pos, neg.drop(columns="source_peak"), seed=0, mode="paired")
    assert summary["mode"] == "random"
    assert len(set(groups["test"]["pair"])) == len(data["test"][0])   # every sequence its own unit
