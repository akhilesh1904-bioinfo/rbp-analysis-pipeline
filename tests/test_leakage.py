"""Sequence leakage between training and val/test sets."""
import random

import pandas as pd

from rbp_pipeline.genome import revcomp
from rbp_pipeline.leakage import check_leakage, find_leaks, kmer_set

rng = random.Random(0)


def rand(n):
    return "".join(rng.choice("ACGT") for _ in range(n))


def frames(train_pos, test_pos, train_neg=(), test_neg=()):
    pos = pd.DataFrame({"peak_id": [f"p{i}" for i in range(len(train_pos) + len(test_pos))],
                        "sequence": list(train_pos) + list(test_pos),
                        "split": ["train"] * len(train_pos) + ["test"] * len(test_pos)})
    neg = pd.DataFrame({"neg_id": [f"n{i}" for i in range(len(train_neg) + len(test_neg))],
                        "sequence": list(train_neg) + list(test_neg),
                        "split": ["train"] * len(train_neg) + ["test"] * len(test_neg)})
    return pos, neg


def test_detects_exact_revcomp_and_shared_kmer():
    a, b, c = rand(100), rand(100), rand(100)
    shared = rand(60) + c[20:45] + rand(15)          # 25 bp in common with c
    pos, neg = frames([a, b, c], [a, revcomp(b), shared, rand(100)])
    leaks = find_leaks(pos, neg, k=20).set_index("seq_id")["reason"].to_dict()
    assert leaks == {"p3": "exact", "p4": "revcomp", "p5": "shared_kmer"}


def test_negatives_are_checked_too():
    a = rand(100)
    pos, neg = frames([rand(100)], [rand(100)], train_neg=[a], test_neg=[a])
    leaks = find_leaks(pos, neg, k=20)
    assert list(leaks["seq_id"]) == ["n1"] and list(leaks["label"]) == [0]


def test_low_complexity_kmers_ignored():
    assert kmer_set(["A" * 30, "AT" * 15], 20) == set()
    pos, neg = frames([rand(40) + "A" * 30], [rand(40) + "A" * 30])
    assert find_leaks(pos, neg, k=20).empty


def test_remove_mode_drops_flagged_sequences():
    a = rand(100)
    pos, neg = frames([a, rand(100)], [a, rand(100)])
    pos2, neg2, summary, leaks = check_leakage(pos, neg, "remove", 20)
    assert "p2" not in set(pos2["peak_id"]) and len(pos2) == 3
    assert summary["test"]["n_flagged"] == 1 and summary["mode"] == "remove"
    pos3, _, _, _ = check_leakage(pos, neg, "report", 20)
    assert len(pos3) == 4
