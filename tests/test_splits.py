"""Chromosome split correctness and reproducibility (bug 5: seeding)."""
import numpy as np
import pandas as pd
import pytest
import torch

import run_rbp_analysis as rbp


def make_df(chroms, prefix):
    return pd.DataFrame({"chrom": chroms, "sequence": [f"{prefix}{i}" for i in range(len(chroms))]})


def test_splits_are_chromosome_disjoint_and_balanced():
    chroms = ["chr1"] * 10 + ["chr19"] * 4 + ["chr21"] * 3
    pos = make_df(chroms, "P")
    neg = make_df(chroms + ["chr21"] * 5, "N")
    data, pos_out, neg_out = rbp.build_splits(pos, neg, ["chr21"], ["chr19"])
    assert set(pos_out.loc[pos_out.split == "test", "chrom"]) == {"chr21"}
    assert set(pos_out.loc[pos_out.split == "val", "chrom"]) == {"chr19"}
    for split, (seqs, labels) in data.items():
        assert sum(labels) * 2 == len(labels)       # balanced
    test_seqs = set(data["test"][0])
    assert test_seqs.isdisjoint(data["train"][0])


def test_overlapping_test_and_val_rejected():
    pos = make_df(["chr1", "chr21"], "P")
    with pytest.raises(SystemExit, match="both test and val"):
        rbp.build_splits(pos, pos, ["chr21"], ["chr21"])


def test_empty_val_rejected():
    pos = make_df(["chr1", "chr21"], "P")
    with pytest.raises(SystemExit, match="val set is empty"):
        rbp.build_splits(pos, pos, ["chr21"], ["chr19"])


def test_cnn_training_is_reproducible(tmp_path):
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), 40)) for _ in range(64)]
    labels = [i % 2 for i in range(64)]
    rbp.MAX_LEN = 40
    weights = []
    for run in range(2):
        model = rbp.train_cnn(seqs, labels, seqs[:16], labels[:16], tmp_path, f"r{run}", max_epochs=2)
        weights.append(model.state_dict())
    for k in weights[0]:
        assert torch.equal(weights[0][k], weights[1][k]), k
