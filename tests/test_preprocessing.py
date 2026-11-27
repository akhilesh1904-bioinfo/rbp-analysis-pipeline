"""BED validation: chromosome naming, empty input, significance filters, windowing, strand-aware dedup."""
import pandas as pd
import pytest

from rbp_pipeline import PipelineError
from rbp_pipeline.genome import normalise_chrom
from rbp_pipeline.preprocessing import apply_window, read_bed, validate_bed


@pytest.mark.parametrize("raw,expected", [
    ("chr1", "chr1"), ("1", "chr1"), ("X", "chrX"), ("MT", "chrM"), ("chrX", "chrX"),
])
def test_normalise_chrom(raw, expected):
    assert normalise_chrom(raw) == expected


def write(tmp_path, lines):
    p = tmp_path / "in.bed"
    p.write_text("\n".join("\t".join(map(str, l)) for l in lines) + "\n")
    return p


def test_ensembl_names_are_accepted(tmp_path):
    df = validate_bed(write(tmp_path, [("1", 10, 50, "a", 0, "+"), ("X", 10, 50, "b", 0, "-")]), tmp_path)
    assert list(df["chrom"]) == ["chr1", "chrX"]


def test_no_valid_peaks_gives_clear_error(tmp_path):
    with pytest.raises(PipelineError, match="No usable peaks"):
        validate_bed(write(tmp_path, [("scaffold_1", 10, 50), ("chr1", 50, 10)]), tmp_path)


def test_encode_significance_filters(tmp_path):
    bed = write(tmp_path, [
        ("chr1", 10, 50, "keep", 1000, "+", 4.0, 5.0),
        ("chr1", 60, 90, "lowfc", 1000, "+", 1.0, 5.0),
        ("chr1", 100, 130, "lowp", 1000, "+", 4.0, 1.0),
    ])
    assert list(validate_bed(bed, tmp_path, min_log2fc=3, min_log10p=3)["name"]) == ["keep"]


def test_same_interval_on_both_strands_is_kept(tmp_path):
    df = validate_bed(write(tmp_path, [("chr1", 10, 50, "a", 0, "+"), ("chr1", 10, 50, "b", 0, "-"),
                                       ("chr1", 10, 50, "dup", 0, "+")]), tmp_path)
    assert sorted(df["strand"]) == ["+", "-"]
    assert df["peak_id"].is_unique


def test_cleaned_bed_uses_peak_ids(tmp_path):
    validate_bed(write(tmp_path, [("chr1", 10, 50, "a", 0, "+")]), tmp_path)
    assert (tmp_path / "peaks_cleaned.bed").read_text().split("\t")[3] == "peak_0"


def test_apply_window_centres_and_clips():
    out = apply_window(pd.DataFrame({"start": [1000, 10], "end": [1004, 20]}), 100)
    assert list(out["start"]) == [952, 0]
    assert (out["end"] - out["start"] == 100).all()


def test_window_applied_in_validation(tmp_path):
    bed = tmp_path / "p.bed"
    bed.write_text("chr1\t1000\t1002\ta\t0\t+\nchr1\t5000\t5300\tb\t0\t-\n")
    df = validate_bed(bed, tmp_path, window=50)
    assert (df["peak_length"] == 50).all()
    assert list(df["orig_end"] - df["orig_start"]) == [2, 300]


def test_read_bed_keeps_source_peak(tmp_path):
    bed = tmp_path / "n.bed"
    bed.write_text("chr1\t0\t10\tpeak_3\t0\t-\n1\t5\t15\n")
    df = read_bed(bed, "neg")
    assert list(df["source_peak"]) == ["peak_3", "."]
    assert list(df["chrom"]) == ["chr1", "chr1"] and list(df["neg_id"]) == ["neg_0", "neg_1"]
