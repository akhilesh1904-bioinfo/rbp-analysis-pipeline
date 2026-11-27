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
