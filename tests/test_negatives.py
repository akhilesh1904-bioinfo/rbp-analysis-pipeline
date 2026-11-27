"""Negative generation: GC matching, strand-aware and region-matched shuffling, region QC."""
import pandas as pd
import pytest

from conftest import needs_tools, write_fasta
from rbp_pipeline.genome import Genome
from rbp_pipeline.negative_sampling import generate_candidates, region_qc, select_gc_matched


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


@pytest.fixture
def shuffle_setup(tmp_path):
    genome = Genome(write_fasta(tmp_path / "g.fa", {"chr1": "ACGT" * 5000}))
    peaks = tmp_path / "peaks.bed"
    rows = [f"chr1\t{1000 + 200 * i}\t{1050 + 200 * i}\tp{i}\t0\t{'+-'[i % 2]}\n" for i in range(20)]
    peaks.write_text("".join(rows))
    return genome, peaks, tmp_path


def _read(path):
    return [l.split("\t") for l in path.read_text().splitlines() if l.strip()]


def _in(regions, row):
    s, e = int(row[1]), int(row[2])
    return any(r[0] == row[0] and int(r[1]) <= s and e <= int(r[2]) for r in regions)


@needs_tools
def test_strand_aware_shuffle_keeps_negatives_on_same_strand_regions(shuffle_setup):
    genome, peaks, tmp = shuffle_setup
    regions = tmp / "regions.bed"
    plus, minus = ("chr1", "5000", "10000"), ("chr1", "12000", "19000")
    regions.write_text(f"{chr(9).join(plus)}\tx\t0\t+\n{chr(9).join(minus)}\tx\t0\t-\n")
    stats = generate_candidates(peaks, genome, regions, tmp / "neg.bed", rounds=5, seed=1)
    rows = _read(tmp / "neg.bed")
    assert stats["stranded"] == 1 and len(rows) == 100
    for r in rows:
        assert _in([plus] if r[5] == "+" else [minus], r), r


@needs_tools
def test_unstranded_regions_warn_and_fall_back(shuffle_setup, caplog):
    genome, peaks, tmp = shuffle_setup
    regions = tmp / "regions.bed"
    regions.write_text("chr1\t5000\t19000\n")
    stats = generate_candidates(peaks, genome, regions, tmp / "neg.bed", rounds=1, seed=1)
    assert stats["stranded"] == 0 and len(_read(tmp / "neg.bed")) == 20
    assert "antisense" in caplog.text


@needs_tools
def test_region_matched_shuffle(shuffle_setup):
    genome, peaks, tmp = shuffle_setup
    utr, intron = tmp / "utr.bed", tmp / "intron.bed"
    utr.write_text("chr1\t5000\t9000\tUTR\t0\t+\nchr1\t5000\t9000\tUTR\t0\t-\n")
    intron.write_text("chr1\t12000\t19000\tintronic\t0\t+\n")            # no '-' introns
    groups = {f"p{i}": ("UTR" if i < 10 else "intronic") for i in range(20)}
    groups["p0"] = "intergenic"                                                  # not a class -> all
    stats = generate_candidates(peaks, genome, {"UTR": utr, "intronic": intron}, tmp / "neg.bed",
                                rounds=3, seed=2, peak_groups=groups)
    rows = _read(tmp / "neg.bed")
    u = [("chr1", "5000", "9000")]
    i_ = [("chr1", "12000", "19000")]
    for r in rows:
        pid, strand = r[3], r[5]
        n = int(pid[1:])
        if pid == "p0":
            continue
        if n < 10:
            assert _in(u, r), r                       # UTR peaks stay in UTRs
        elif strand == "+":
            assert _in(i_, r), r                      # '+' intronic peaks stay in introns
        else:
            assert _in(u, r), r                       # '-' intronic peaks: no '-' introns -> fall back
    assert stats["n_fallback"] == 1 + 5               # p0 + the five '-' intronic peaks


def test_region_qc_flags_antisense_and_region_mismatch(caplog):
    pos = pd.DataFrame({"region_type": ["UTR"] * 7 + ["exonic"] * 3, "antisense": [False] * 10})
    neg = pd.DataFrame({"region_type": ["UTR"] * 2 + ["exonic"] * 4 + ["intergenic"] * 4,
                        "antisense": [False] * 6 + [True] * 3 + [False]})
    qc = region_qc(pos, neg)
    assert qc["negatives"]["antisense"] == pytest.approx(0.3)
    assert qc["distance"] == pytest.approx(0.5)
    assert len(qc["warnings"]) == 2 and "antisense" in qc["warnings"][0]
    good = region_qc(pos, pos)
    assert good["distance"] == 0 and not good["warnings"] and good["class_match"] is None
    pos2 = pos.assign(peak_id=[f"p{i}" for i in range(10)])
    neg2 = pos.assign(source_peak=[f"p{i}" for i in range(10)])
    assert region_qc(pos2, neg2)["class_match"] == 1.0
