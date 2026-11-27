"""Strand-aware sequence extraction, row alignment, and reference naming checks."""
import shutil

import pandas as pd
import pytest

from conftest import write_fasta
from rbp_pipeline import PipelineError
from rbp_pipeline.genome import Genome, check_reference_naming, clean_seq, gc_fraction, revcomp
from rbp_pipeline.sequence_extraction import attach_sequences

# chr1 = AAAACCCCGGGGTTTT
REGIONS = pd.DataFrame({
    "chrom": ["chr1", "chrUn_missing", "chr1", "chr1"],
    "start": [0, 0, 0, 4], "end": [8, 4, 8, 12],
    "strand": ["+", "+", "-", "-"],
    "peak_id": ["plus", "missing", "minus", "minus2"],
})
EXPECTED = {"plus": "AAAACCCC", "minus": "GGGGTTTT", "minus2": "CCCCGGGG"}


def test_revcomp_and_clean():
    assert revcomp("AACGTN") == "NACGTT"
    assert clean_seq("acgu-xn") == "ACGTNNN"


def test_gc_fraction_ignores_n():
    assert gc_fraction("GGCCNNNN") == 1.0 and gc_fraction("AATT") == 0.0 and gc_fraction("NNNN") == 0.0


def _extract(genome, tmp_path, fetched):
    out = attach_sequences(REGIONS, fetched, "peak_id", tmp_path / "out.fa")
    return dict(zip(out["peak_id"], out["sequence"]))


@pytest.mark.skipif(shutil.which("samtools") is None, reason="samtools not installed")
def test_samtools_extraction_is_strand_aware_and_aligned(tiny_genome, tmp_path):
    g = Genome(tiny_genome)
    fetched = g._fetch_samtools(REGIONS, tmp_path)
    assert fetched is not None
    assert _extract(g, tmp_path, fetched) == EXPECTED


def test_in_memory_extraction_is_strand_aware_and_aligned(tiny_genome, tmp_path):
    g = Genome(tiny_genome)
    assert _extract(g, tmp_path, g._fetch_in_memory(REGIONS)) == EXPECTED


def test_mostly_n_sequences_dropped(tiny_genome, tmp_path):
    df = pd.DataFrame({"chrom": ["chr2", "chr2"], "start": [0, 4], "end": [4, 12],
                       "strand": ["+", "+"], "peak_id": ["ok", "allN"]})
    out = attach_sequences(df, Genome(tiny_genome)._fetch_in_memory(df), "peak_id", tmp_path / "o.fa")
    assert list(out["peak_id"]) == ["ok"]


def test_fasta_headers_carry_ids_and_strand(tiny_genome, tmp_path):
    attach_sequences(REGIONS, Genome(tiny_genome)._fetch_in_memory(REGIONS), "peak_id", tmp_path / "o.fa")
    headers = [l for l in (tmp_path / "o.fa").read_text().splitlines() if l.startswith(">")]
    assert ">minus::chr1:1-8(-)" in headers


def test_ensembl_named_genome_rejected(tmp_path):
    g = Genome(write_fasta(tmp_path / "ens.fa", {"1": "ACGT" * 10}))
    with pytest.raises(PipelineError, match="Ensembl-style"):
        check_reference_naming(g, None)


def test_scaffolds_sorted_first_are_ignored(tiny_genome, tmp_path):
    """Regression: GENCODE scaffolds (GL000..., KI270...) sort before chr1 and must not trigger an error."""
    bed = tmp_path / "exons.bed"
    bed.write_text("GL000008.2\t0\t100\nKI270706.1\t0\t50\nchr1\t0\t10\n")
    check_reference_naming(Genome(tiny_genome), bed)


def test_unprefixed_region_bed_rejected(tiny_genome, tmp_path):
    bed = tmp_path / "genes.bed"
    bed.write_text("1\t0\t100\n2\t0\t50\n")
    with pytest.raises(PipelineError, match="without the 'chr' prefix"):
        check_reference_naming(Genome(tiny_genome), bed)
