"""Strand-aware region annotation via bedtools."""
import shutil

import pandas as pd
import pytest

from rbp_pipeline.annotation import (annotate_regions, bed_is_stranded, feature_bed_from_gtf,
                                     protein_coding_genes, region_beds_from_gtf, region_fractions)

pytestmark = pytest.mark.skipif(shutil.which("bedtools") is None, reason="bedtools required")

GTF = (
    'chr1\ts\tgene\t1\t10000\t.\t+\t.\tgene_id "g1"; gene_name "PLUSGENE"; gene_type "protein_coding";\n'
    'chr1\ts\tUTR\t1\t500\t.\t+\t.\tgene_id "g1"; gene_name "PLUSGENE";\n'          # shares start with gene
    'chr1\ts\texon\t1\t2000\t.\t+\t.\tgene_id "g1"; gene_name "PLUSGENE";\n'
    'chr1\ts\texon\t1\t2000\t.\t+\t.\tgene_id "g1"; gene_name "PLUSGENE";\n'        # duplicate (2nd transcript)
    'chr1\ts\tCDS\t501\t2000\t.\t+\t.\tgene_id "g1"; gene_name "PLUSGENE";\n'
    'chr1\ts\tgene\t20001\t30000\t.\t-\t.\tgene_id "g2"; gene_name "MINUSGENE"; gene_type "lncRNA";\n'
    'GL000008.2\ts\tgene\t1\t100\t.\t+\t.\tgene_id "g3"; gene_name "SCAFFOLD";\n'
)


@pytest.fixture
def gtf(tmp_path):
    p = tmp_path / "a.gtf"
    p.write_text(GTF)
    return p


def peaks(rows):
    return pd.DataFrame(rows, columns=["peak_id", "chrom", "start", "end", "strand"])


def test_feature_index_keeps_utr_sharing_gene_start_and_drops_duplicates(gtf, tmp_path):
    """Regression: `sort -u` on key fields only silently dropped UTRs that start where their gene starts."""
    lines = feature_bed_from_gtf(gtf, tmp_path / "cache").read_text().splitlines()
    assert "chr1\t0\t500\tPLUSGENE\tUTR\t+\t." in lines
    assert "chr1\t0\t10000\tPLUSGENE\tintronic\t+\tprotein_coding" in lines
    assert sum("\texonic\t" in l and l.startswith("chr1\t0\t2000") for l in lines) == 1
    assert not any(l.startswith("GL000008") for l in lines)


def test_strand_aware_classes_and_priority(gtf, tmp_path):
    df = peaks([
        ("utr", "chr1", 100, 200, "+"),        # UTR beats exon/gene
        ("cds", "chr1", 1000, 1100, "+"),      # exonic
        ("intron", "chr1", 5000, 5100, "+"),   # inside gene, no exon
        ("antisense", "chr1", 5000, 5100, "-"),  # + gene only: not this peak's gene
        ("minus", "chr1", 25000, 25100, "-"),  # intronic in the - gene
        ("nostrand", "chr1", 5000, 5100, "."),  # matches either strand
        ("far", "chr2", 0, 100, "+"),
    ])
    out = annotate_regions(df, gtf, tmp_path, tmp_path / "cache").set_index("peak_id")
    assert out.loc["utr", "region_type"] == "UTR"
    assert out.loc["cds", "region_type"] == "exonic"
    assert out.loc["intron", ["region_type", "gene_name"]].tolist() == ["intronic", "PLUSGENE"]
    assert out.loc["antisense", "region_type"] == "intergenic"
    assert out.loc["minus", ["region_type", "gene_name"]].tolist() == ["intronic", "MINUSGENE"]
    assert out.loc["nostrand", "region_type"] == "intronic"
    assert out.loc["far", ["region_type", "gene_name"]].tolist() == ["intergenic", "."]
    # antisense: no gene on its own strand, but one on the other
    assert out.loc["antisense", "antisense"] and not out.loc["far", "antisense"] and not out.loc["intron", "antisense"]
    assert out.loc["intron", "gene_type"] == "protein_coding"
    assert out.loc["minus", "gene_type"] == "lncRNA"
    fr = region_fractions(out.reset_index())
    assert fr["antisense"] == pytest.approx(1 / 7) and fr["intergenic"] == pytest.approx(2 / 7)


def test_annotate_with_other_id_column(gtf, tmp_path):
    df = peaks([("n1", "chr1", 100, 200, "+")]).rename(columns={"peak_id": "neg_id"})
    out = annotate_regions(df, gtf, tmp_path, tmp_path / "cache", id_col="neg_id", quiet=True)
    assert out.loc[0, "region_type"] == "UTR"


def read_bed(path):
    return [l.split("\t") for l in path.read_text().splitlines()]


def test_region_beds_are_stranded_disjoint_and_follow_priority(gtf, tmp_path):
    beds = region_beds_from_gtf(gtf, tmp_path / "cache")
    utr, exonic, intronic = (read_bed(beds[c]) for c in ("UTR", "exonic", "intronic"))
    assert [r[:3] + [r[5]] for r in utr] == [["chr1", "0", "500", "+"]]
    assert [r[:3] + [r[5]] for r in exonic] == [["chr1", "500", "2000", "+"]]       # exon minus UTR
    assert sorted((r[1], r[2], r[5]) for r in intronic) == [("2000", "10000", "+"), ("20000", "30000", "-")]
    assert all(bed_is_stranded(p) for p in beds.values())
    mtime = beds["UTR"].stat().st_mtime_ns
    assert region_beds_from_gtf(gtf, tmp_path / "cache")["UTR"].stat().st_mtime_ns == mtime   # cached


def test_bed_is_stranded(tmp_path):
    a, b, c = tmp_path / "a.bed", tmp_path / "b.bed", tmp_path / "c.bed"
    a.write_text("chr1\t0\t10\tx\t0\t+\nchr1\t20\t30\tx\t0\t-\n")
    b.write_text("chr1\t0\t10\nchr1\t20\t30\n")
    c.write_text("chr1\t0\t10\tx\t0\t+\nchr1\t20\t30\tx\t0\t.\n")
    assert bed_is_stranded(a) and not bed_is_stranded(b) and not bed_is_stranded(c)


def test_protein_coding_genes(gtf, tmp_path):
    assert protein_coding_genes(gtf, tmp_path / "cache") == {"PLUSGENE"}


def test_feature_cache_is_reused(gtf, tmp_path):
    first = feature_bed_from_gtf(gtf, tmp_path / "cache")
    mtime = first.stat().st_mtime_ns
    assert feature_bed_from_gtf(gtf, tmp_path / "cache").stat().st_mtime_ns == mtime
