"""BED validation: chromosome naming, empty input, significance filters, strand-aware dedup."""
import pytest

import run_rbp_analysis as rbp


@pytest.mark.parametrize("raw,expected", [
    ("chr1", "chr1"), ("1", "chr1"), ("X", "chrX"), ("MT", "chrM"), ("chrX", "chrX"),
])
def test_normalise_chrom(raw, expected):
    assert rbp.normalise_chrom(raw) == expected


def write(tmp_path, lines):
    p = tmp_path / "in.bed"
    p.write_text("\n".join("\t".join(map(str, l)) for l in lines) + "\n")
    return p


def test_ensembl_names_are_accepted(tmp_path):
    bed = write(tmp_path, [("1", 10, 50, "a", 0, "+"), ("X", 10, 50, "b", 0, "-")])
    df = rbp.validate_bed(bed, tmp_path)
    assert list(df["chrom"]) == ["chr1", "chrX"]


def test_no_valid_peaks_gives_clear_error(tmp_path):
    bed = write(tmp_path, [("scaffold_1", 10, 50), ("chr1", 50, 10)])
    with pytest.raises(SystemExit, match="no usable peaks"):
        rbp.validate_bed(bed, tmp_path)


def test_encode_significance_filters(tmp_path):
    bed = write(tmp_path, [
        ("chr1", 10, 50, "keep", 1000, "+", 4.0, 5.0),
        ("chr1", 60, 90, "lowfc", 1000, "+", 1.0, 5.0),
        ("chr1", 100, 130, "lowp", 1000, "+", 4.0, 1.0),
    ])
    df = rbp.validate_bed(bed, tmp_path, min_log2fc=3, min_log10p=3)
    assert list(df["name"]) == ["keep"]


def test_same_interval_on_both_strands_is_kept(tmp_path):
    bed = write(tmp_path, [("chr1", 10, 50, "a", 0, "+"), ("chr1", 10, 50, "b", 0, "-"),
                           ("chr1", 10, 50, "dup", 0, "+")])
    df = rbp.validate_bed(bed, tmp_path)
    assert sorted(df["strand"]) == ["+", "-"]
    assert df["peak_id"].is_unique


def test_cleaned_bed_uses_peak_ids(tmp_path):
    bed = write(tmp_path, [("chr1", 10, 50, "a", 0, "+")])
    rbp.validate_bed(bed, tmp_path)
    cols = (tmp_path / "peaks_cleaned.bed").read_text().split("\t")
    assert cols[3] == "peak_0"
