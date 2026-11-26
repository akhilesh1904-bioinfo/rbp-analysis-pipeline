import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_rbp_analysis as rbp  # noqa: E402

MOTIF = "TGCATGTGCATG"   # planted in the RNA sense of every synthetic peak


@pytest.fixture(autouse=True)
def restore_globals():
    """configure() mutates module globals; reset them after every test."""
    saved = {k: getattr(rbp, k) for k in ("GENOME_FA", "GENES_BED", "GTF_FILE", "MAX_LEN", "RANDOM_STATE")}
    yield
    for k, v in saved.items():
        setattr(rbp, k, v)


def write_fasta(path: Path, chroms: dict, width: int = 60) -> Path:
    with open(path, "w") as f:
        for name, seq in chroms.items():
            f.write(f">{name}\n")
            for i in range(0, len(seq), width):
                f.write(seq[i:i + width] + "\n")
    return path


@pytest.fixture
def tiny_genome(tmp_path):
    """chr1 = AAAACCCCGGGGTTTT so strand handling is easy to check by eye."""
    return write_fasta(tmp_path / "tiny.fa", {"chr1": "AAAACCCCGGGGTTTT", "chr2": "ACGTNNNNNNNNACGT"})


@pytest.fixture
def synthetic_dataset(tmp_path):
    """
    A small genome where every peak carries MOTIF on its own strand
    (reverse-complemented in the genome for '-' peaks).
    """
    rng = random.Random(0)
    chrom_names = ["chr1", "chr2", "chr3", "chr4", "chr19", "chr20", "chr21", "chr22", "chrX"]
    length = 40_000
    genome = {c: [rng.choice("ACGT") for _ in range(length)] for c in chrom_names}

    peaks = []
    for c in chrom_names:
        for i in range(45):
            start = 1_000 + i * 800
            plen = rng.randint(60, 90)
            strand = rng.choice("+-")
            site = rbp.revcomp(MOTIF) if strand == "-" else MOTIF
            pos = start + (plen - len(MOTIF)) // 2
            genome[c][pos:pos + len(MOTIF)] = list(site)
            peaks.append((c, start, start + plen, f"p{c}_{i}", 1000, strand, 4.0, 5.0))

    genome = {c: "".join(s) for c, s in genome.items()}
    fasta = write_fasta(tmp_path / "genome.fa", genome)

    genes_bed = tmp_path / "genes.bed"
    genes_bed.write_text("".join(f"{c}\t0\t{length}\tgene_{c}\t0\t+\n" for c in chrom_names))

    peaks_bed = tmp_path / "peaks.bed"
    peaks_bed.write_text("".join("\t".join(map(str, p)) + "\n" for p in peaks))
    return {"genome": fasta, "genes_bed": genes_bed, "peaks_bed": peaks_bed,
            "genome_seqs": genome, "tmp": tmp_path}
