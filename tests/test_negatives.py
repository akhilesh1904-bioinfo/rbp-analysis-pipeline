"""GC-matched negative selection and fixed-window resizing."""
import pandas as pd

import run_rbp_analysis as rbp


def test_gc_fraction_ignores_n():
    assert rbp.gc_fraction("GGCCNNNN") == 1.0
    assert rbp.gc_fraction("AATT") == 0.0
    assert rbp.gc_fraction("NNNN") == 0.0


def test_select_gc_matched_picks_closest_candidate():
    pos = pd.DataFrame({"peak_id": ["p1", "p2"], "sequence": ["GGGGCCCC", "AAAATTTT"]})
    cand = pd.DataFrame({
        "chrom": ["chr1"] * 4, "start": [0, 100, 200, 300], "end": [8, 108, 208, 308],
        "strand": ["+"] * 4,
        "source_peak": ["p1", "p1", "p2", "p2"],
        "neg_id": ["n0", "n1", "n2", "n3"],
        "sequence": ["AAAAAAAA", "GGGCCCCA", "GGGGGGGG", "AAATTTTA"],
    })
    out = rbp.select_gc_matched(pos, cand)
    assert dict(zip(out["source_peak"], out["neg_id"])) == {"p1": "n1", "p2": "n3"}


def test_select_gc_matched_never_reuses_an_interval():
    pos = pd.DataFrame({"peak_id": ["p1", "p2"], "sequence": ["GGCC", "GGCC"]})
    cand = pd.DataFrame({
        "chrom": ["chr1"] * 3, "start": [0, 0, 50], "end": [4, 4, 54], "strand": ["+"] * 3,
        "source_peak": ["p1", "p2", "p2"], "neg_id": ["n0", "n1", "n2"],
        "sequence": ["GGCC", "GGCC", "GGCA"],
    })
    out = rbp.select_gc_matched(pos, cand)
    assert len(out) == 2
    assert not out.duplicated(["chrom", "start", "end"]).any()


def test_apply_window_centres_and_clips():
    df = pd.DataFrame({"start": [1000, 10], "end": [1004, 20]})
    out = rbp.apply_window(df, 100)
    assert list(out["start"]) == [952, 0]
    assert (out["end"] - out["start"] == 100).all()


def test_window_applied_in_validation(tmp_path):
    bed = tmp_path / "p.bed"
    bed.write_text("chr1\t1000\t1002\ta\t0\t+\nchr1\t5000\t5300\tb\t0\t-\n")
    df = rbp.validate_bed(bed, tmp_path, window=50)
    assert (df["peak_length"] == 50).all()
    assert list(df["orig_end"] - df["orig_start"]) == [2, 300]
