"""Bug 2 (strand) and bug 4 (row alignment) in sequence extraction."""
import shutil

import pandas as pd
import pytest

import run_rbp_analysis as rbp

# chr1 = AAAACCCCGGGGTTTT
REGIONS = pd.DataFrame({
    "chrom":   ["chr1", "chrMISSING_IS_NOT_VALID", "chr1", "chr1"],
    "start":   [0, 0, 0, 4],
    "end":     [8, 4, 8, 12],
    "strand":  ["+", "+", "-", "-"],
    "peak_id": ["plus", "missing", "minus", "minus2"],
})
EXPECTED = {"plus": "AAAACCCC", "minus": "GGGGTTTT", "minus2": "CCCCGGGG"}


def test_revcomp():
    assert rbp.revcomp("AACGTN") == "NACGTT"


def test_clean_seq_keeps_positions():
    assert rbp.clean_seq("acgu-xn") == "ACGTNNN"


def _run(tmp_path, fetched):
    out = rbp.attach_sequences(REGIONS, fetched, "peak_id", tmp_path / "out.fa")
    return dict(zip(out["peak_id"], out["sequence"]))


@pytest.mark.skipif(shutil.which("samtools") is None, reason="samtools not installed")
def test_samtools_extraction_is_strand_aware_and_aligned(tiny_genome, tmp_path):
    rbp.GENOME_FA = tiny_genome
    fetched = rbp.fetch_via_samtools(REGIONS, tmp_path)
    assert fetched is not None
    assert _run(tmp_path, fetched) == EXPECTED


def test_in_memory_extraction_is_strand_aware_and_aligned(tiny_genome, tmp_path):
    rbp.GENOME_FA = tiny_genome
    assert _run(tmp_path, rbp.fetch_in_memory(REGIONS)) == EXPECTED


def test_mostly_n_sequences_dropped(tiny_genome, tmp_path):
    rbp.GENOME_FA = tiny_genome
    df = pd.DataFrame({"chrom": ["chr2", "chr2"], "start": [0, 4], "end": [4, 12],
                       "strand": ["+", "+"], "peak_id": ["ok", "allN"]})
    out = rbp.attach_sequences(df, rbp.fetch_in_memory(df), "peak_id", tmp_path / "o.fa")
    assert list(out["peak_id"]) == ["ok"]


def test_fasta_headers_carry_ids_and_strand(tiny_genome, tmp_path):
    rbp.GENOME_FA = tiny_genome
    rbp.attach_sequences(REGIONS, rbp.fetch_in_memory(REGIONS), "peak_id", tmp_path / "o.fa")
    headers = [l for l in (tmp_path / "o.fa").read_text().splitlines() if l.startswith(">")]
    assert ">minus::chr1:1-8(-)" in headers


def test_centre_crop():
    assert rbp.fit_length("AAACCCGGG", 3) == "CCC"
    assert rbp.fit_length("ACG", 10) == "ACG"


def test_one_hot_uses_centre_and_ignores_n():
    rbp.MAX_LEN = 4
    arr = rbp.one_hot("TTACGNTT")      # centre 4 bp = "ACGN"
    assert arr[0, 0] == 1 and arr[1, 1] == 1 and arr[2, 2] == 1 and arr[:, 3].sum() == 0


def test_kmers_skip_n():
    assert "N" not in rbp.seq_to_kmers("ACNGT")
