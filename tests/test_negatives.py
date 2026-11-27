"""GC-matched negative selection."""
import pandas as pd

from rbp_pipeline.negative_sampling import select_gc_matched


def test_select_gc_matched_picks_closest_candidate():
    pos = pd.DataFrame({"peak_id": ["p1", "p2"], "sequence": ["GGGGCCCC", "AAAATTTT"]})
    cand = pd.DataFrame({
        "chrom": ["chr1"] * 4, "start": [0, 100, 200, 300], "end": [8, 108, 208, 308],
        "strand": ["+"] * 4, "source_peak": ["p1", "p1", "p2", "p2"],
        "neg_id": ["n0", "n1", "n2", "n3"],
        "sequence": ["AAAAAAAA", "GGGCCCCA", "GGGGGGGG", "AAATTTTA"],
    })
    out = select_gc_matched(pos, cand)
    assert dict(zip(out["source_peak"], out["neg_id"])) == {"p1": "n1", "p2": "n3"}


def test_select_gc_matched_never_reuses_an_interval():
    pos = pd.DataFrame({"peak_id": ["p1", "p2"], "sequence": ["GGCC", "GGCC"]})
    cand = pd.DataFrame({
        "chrom": ["chr1"] * 3, "start": [0, 0, 50], "end": [4, 4, 54], "strand": ["+"] * 3,
        "source_peak": ["p1", "p2", "p2"], "neg_id": ["n0", "n1", "n2"],
        "sequence": ["GGCC", "GGCC", "GGCA"],
    })
    out = select_gc_matched(pos, cand)
    assert len(out) == 2
    assert not out.duplicated(["chrom", "start", "end"]).any()
