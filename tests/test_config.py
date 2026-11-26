"""Bug 1: --genome / --genes-bed / --gtf were parsed but never used."""
import pytest

import run_rbp_analysis as rbp


def test_configure_sets_reference_paths(tiny_genome, tmp_path):
    genes = tmp_path / "genes.bed"; genes.write_text("chr1\t0\t16\n")
    gtf = tmp_path / "a.gtf"; gtf.write_text("")
    args = rbp.parse_args(["--input", "x.bed", "--name", "P", "--genome", str(tiny_genome),
                           "--genes-bed", str(genes), "--gtf", str(gtf), "--max-len", "50", "--seed", "7"])
    rbp.configure(args)
    assert rbp.GENOME_FA == tiny_genome
    assert rbp.GENES_BED == genes
    assert rbp.GTF_FILE == gtf
    assert rbp.MAX_LEN == 50
    assert rbp.RANDOM_STATE == 7


def test_gtf_is_optional(tiny_genome, tmp_path):
    genes = tmp_path / "genes.bed"; genes.write_text("chr1\t0\t16\n")
    rbp.configure(rbp.parse_args(["--input", "x", "--name", "P", "--genome", str(tiny_genome),
                                  "--genes-bed", str(genes)]))
    assert rbp.GTF_FILE is None


def test_missing_genome_fails_clearly(tmp_path):
    args = rbp.parse_args(["--input", "x", "--name", "P", "--genome", str(tmp_path / "nope.fa"),
                           "--genes-bed", "g.bed"])
    with pytest.raises(SystemExit, match="genome FASTA not found"):
        rbp.configure(args)


def test_genes_bed_required_without_negatives(tiny_genome):
    args = rbp.parse_args(["--input", "x", "--name", "P", "--genome", str(tiny_genome)])
    with pytest.raises(SystemExit, match="--genes-bed is required"):
        rbp.configure(args)


def test_genes_bed_not_needed_with_negatives(tiny_genome, tmp_path):
    neg = tmp_path / "neg.bed"; neg.write_text("chr1\t0\t4\n")
    rbp.configure(rbp.parse_args(["--input", "x", "--name", "P", "--genome", str(tiny_genome),
                                  "--negatives", str(neg)]))
    assert rbp.GENES_BED is None


def test_ensembl_named_genome_rejected(tmp_path):
    from conftest import write_fasta
    rbp.GENOME_FA = write_fasta(tmp_path / "ens.fa", {"1": "ACGT" * 10})
    rbp.GENES_BED = None
    with pytest.raises(SystemExit, match="Ensembl-style"):
        rbp.check_reference_naming()
