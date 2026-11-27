import random
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))      # tests work with or without `pip install -e .`

from rbp_pipeline.genome import revcomp  # noqa: E402

MOTIF = "TGCATGTGCATG"   # planted in the RNA sense of every synthetic peak

needs_tools = pytest.mark.skipif(
    shutil.which("bedtools") is None or shutil.which("samtools") is None,
    reason="bedtools and samtools required")


def write_fasta(path: Path, chroms: dict, width: int = 60) -> Path:
    with open(path, "w") as f:
        for name, seq in chroms.items():
            f.write(f">{name}\n")
            for i in range(0, len(seq), width):
                f.write(seq[i:i + width] + "\n")
    return path


def write_config(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False))
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
            site = revcomp(MOTIF) if strand == "-" else MOTIF
            pos = start + (plen - len(MOTIF)) // 2
            genome[c][pos:pos + len(MOTIF)] = list(site)
            peaks.append((c, start, start + plen, f"p{c}_{i}", 1000, strand, 4.0, 5.0))

    genome = {c: "".join(s) for c, s in genome.items()}
    fasta = write_fasta(tmp_path / "genome.fa", genome)

    genes_bed = tmp_path / "genes.bed"
    genes_bed.write_text("".join(f"{c}\t0\t{length}\tgene_{c}\t0\t+\n" for c in chrom_names))

    peaks_bed = tmp_path / "peaks.bed"
    peaks_bed.write_text("".join("\t".join(map(str, p)) + "\n" for p in peaks))

    gtf = tmp_path / "mini.gtf"
    gtf.write_text(
        'chr21\ts\tgene\t1\t20000\t.\t+\t.\tgene_name "GENEA";\n'
        'chr21\ts\tUTR\t1\t999\t.\t+\t.\tgene_name "GENEA";\n'
        'chr21\ts\texon\t1000\t5000\t.\t+\t.\tgene_name "GENEA";\n')

    return {"genome": fasta, "genes_bed": genes_bed, "peaks_bed": peaks_bed, "gtf": gtf,
            "genome_seqs": genome, "tmp": tmp_path}


@pytest.fixture
def synthetic_config(synthetic_dataset):
    """A YAML config (relative paths) for the synthetic dataset."""
    d = synthetic_dataset
    data = {
        "name": "SYN", "input": "peaks.bed", "output_dir": str(d["tmp"] / "results"),
        "reference": {"genome": "genome.fa", "genes_bed": "genes.bed", "gtf": "mini.gtf"},
        "peaks": {"min_log2fc": 3, "min_log10p": 3},
        "steps": {"report": False},
        "evaluation": {"bootstrap": 50},   # keep tests fast
    }
    return write_config(d["tmp"] / "config.yaml", data), d
