"""Metric calculations, bootstrap intervals and seed summaries."""
import numpy as np
import pytest

from rbp_pipeline.evaluation import (bootstrap_ci, compute_metrics, paired_bootstrap_diff,
                                     summarise_seeds)

Y = [0, 0, 0, 0, 1, 1, 1, 1]
P = [0.1, 0.2, 0.6, 0.3, 0.8, 0.4, 0.9, 0.7]


def test_compute_metrics_known_values():
    m = compute_metrics(Y, P, threshold=0.5)
    assert m["confusion"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 3}
    assert m["precision"] == pytest.approx(0.75) and m["recall"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75) and m["mcc"] == pytest.approx(0.5)
    assert m["roc_auc"] == pytest.approx(15 / 16)   # 15 of 16 pos/neg pairs ranked correctly
    assert 0 < m["brier"] < 0.25


def test_threshold_none_gives_ranking_metrics_only():
    m = compute_metrics(Y, P, threshold=None)
    assert set(m) == {"roc_auc", "pr_auc", "n", "n_pos"}


def test_bootstrap_ci_brackets_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200), np.ones(200)]
    s = y + rng.normal(0, 1, 400)
    ci1 = bootstrap_ci(y, s, "roc_auc", 300, seed=1)
    ci2 = bootstrap_ci(y, s, "roc_auc", 300, seed=1)
    point = compute_metrics(y, s, None)["roc_auc"]
    assert ci1 == ci2 and ci1[0] < point < ci1[1]
    assert np.isnan(bootstrap_ci(y, s, "roc_auc", 0)[0])


def test_paired_difference_detects_better_model():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(300), np.ones(300)]
    good, bad = y + rng.normal(0, 0.5, 600), y + rng.normal(0, 2.0, 600)
    d = paired_bootstrap_diff(y, good, bad, 300, seed=0)
    assert d["diff"] > 0 and d["ci_low"] > 0


def test_summarise_seeds():
    runs = [{"cnn": {"roc_auc": 0.8, "n": 10}}, {"cnn": {"roc_auc": 0.9, "n": 10}}]
    s = summarise_seeds(runs)["cnn"]
    assert "n" not in s
    assert s["roc_auc"]["mean"] == pytest.approx(0.85)
    assert s["roc_auc"]["sd"] == pytest.approx(np.std([0.8, 0.9], ddof=1))


def test_grouped_bootstrap_with_singleton_groups_matches_ungrouped():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(100), np.ones(100)]
    s = y + rng.normal(0, 1, 200)
    groups = [f"g{i}" for i in range(200)]
    assert bootstrap_ci(y, s, "roc_auc", 200, seed=3) == bootstrap_ci(y, s, "roc_auc", 200, seed=3, groups=groups)


def test_grouped_bootstrap_widens_ci_when_sequences_cluster():
    # 20 genes, each contributing 10 near-identical pairs: the effective sample is 20 genes, not 200 pairs.
    rng = np.random.default_rng(1)
    y, s, genes = [], [], []
    for g in range(20):
        shift = rng.normal(0, 1.5)          # gene-level effect shared by all its pairs
        for _ in range(10):
            y += [1, 0]
            s += [1 + shift + rng.normal(0, 0.1), shift + rng.normal(0, 0.1) - 0.2]
            genes += [f"gene{g}", f"gene{g}"]
    flat = bootstrap_ci(y, s, "roc_auc", 400, seed=0)
    grouped = bootstrap_ci(y, s, "roc_auc", 400, seed=0, groups=genes)
    assert (grouped[1] - grouped[0]) > 1.5 * (flat[1] - flat[0])


def test_grouped_paired_difference_and_length_check():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(300), np.ones(300)]
    good, bad = y + rng.normal(0, 0.5, 600), y + rng.normal(0, 2.0, 600)
    pairs = [f"p{i}" for i in range(300)] * 2
    d = paired_bootstrap_diff(y, good, bad, 300, seed=0, groups=pairs)
    assert d["diff"] > 0 and d["ci_low"] > 0
    with pytest.raises(ValueError, match="groups"):
        bootstrap_ci(y, good, "roc_auc", 10, groups=pairs[:10])
