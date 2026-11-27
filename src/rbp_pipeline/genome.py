"""Genome access: chromosome naming, FASTA index, strand-aware sequence fetching."""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from . import PipelineError

log = logging.getLogger("rbp_pipeline")

VALID_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY", "chrM"}

GRCH38_SIZES = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555,
    "chr5": 181538259, "chr6": 170805979, "chr7": 159345973, "chr8": 145138636,
    "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345,
    "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
    "chr21": 46709983, "chr22": 50818468, "chrX": 156040895, "chrY": 57227415,
}

_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def normalise_chrom(chrom: str) -> str:
    """Convert Ensembl-style names (1, X, MT) to UCSC/GENCODE style (chr1, chrX, chrM)."""
    if chrom.startswith("chr"):
        return chrom
    if chrom in ("MT", "M"):
        return "chrM"
    return "chr" + chrom


def clean_seq(seq: str) -> str:
    """Uppercase, U->T, and replace any non-ACGT character with N (keeps positions intact)."""
    return re.sub(r"[^ACGTN]", "N", seq.upper().replace("U", "T"))


def revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def gc_fraction(seq: str) -> float:
    acgt = sum(seq.count(b) for b in "ACGT")
    return (seq.count("G") + seq.count("C")) / acgt if acgt else 0.0


def region_key(chrom: str, start: int, end: int) -> str:
    """samtools-style 1-based inclusive region for a 0-based half-open BED interval."""
    return f"{chrom}:{start + 1}-{end}"


class Genome:
    """A genome FASTA. Uses samtools faidx when available, otherwise reads the FASTA directly."""

    def __init__(self, fasta: Path):
        self.fasta = Path(fasta)
        self._fai: Optional[Path] = None
        self._fai_checked = False

    @property
    def fai(self) -> Optional[Path]:
        """The .fai index, created with samtools if missing. None if it can't be made."""
        if not self._fai_checked:
            self._fai_checked = True
            fai = Path(str(self.fasta) + ".fai")
            if not fai.exists():
                try:
                    subprocess.run(["samtools", "faidx", str(self.fasta)], check=True, capture_output=True)
                except (FileNotFoundError, subprocess.CalledProcessError):
                    pass
            self._fai = fai if fai.exists() else None
        return self._fai

    def chrom_sizes(self) -> Dict[str, int]:
        if self.fai is not None:
            sizes = {}
            with open(self.fai) as f:
                for line in f:
                    p = line.rstrip("\n").split("\t")
                    if len(p) >= 2:
                        sizes[p[0]] = int(p[1])
            return sizes
        return {}

    def chrom_names(self) -> List[str]:
        sizes = self.chrom_sizes()
        if sizes:
            return list(sizes)
        with open(self.fasta) as f:
            return [line[1:].split()[0] for line in f if line.startswith(">")]

    # ── fetching ────────────────────────────────────────────────────────────

    def fetch(self, regions: pd.DataFrame, work_dir: Path) -> Dict[str, str]:
        """Return {region_key: + strand sequence} for every (chrom, start, end) row."""
        fetched = self._fetch_samtools(regions, work_dir)
        if fetched is not None:
            log.info("  Extracted via samtools (fast)")
            return fetched
        log.info("  samtools unavailable — reading the genome FASTA directly (slow for GRCh38)")
        return self._fetch_in_memory(regions)

    def _fetch_samtools(self, regions: pd.DataFrame, work_dir: Path) -> Optional[Dict[str, str]]:
        if self.fai is None:
            return None
        available = set(self.chrom_sizes())
        keys = sorted({region_key(r.chrom, r.start, r.end)
                       for r in regions.itertuples() if r.chrom in available})
        if not keys:
            return {}
        tmp = Path(work_dir) / "_regions_tmp.txt"
        tmp.write_text("\n".join(keys) + "\n")
        try:
            result = subprocess.run(["samtools", "faidx", str(self.fasta), "-r", str(tmp)],
                                    capture_output=True, text=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        finally:
            tmp.unlink(missing_ok=True)
        return _parse_fasta_text(result.stdout.splitlines())

    def _fetch_in_memory(self, regions: pd.DataFrame) -> Dict[str, str]:
        wanted = set(regions["chrom"])
        genome, hdr, buf = {}, None, []
        with open(self.fasta) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if hdr in wanted:
                        genome[hdr] = "".join(buf)
                    hdr, buf = line[1:].split()[0], []
                elif hdr in wanted:
                    buf.append(line)
        if hdr in wanted:
            genome[hdr] = "".join(buf)
        return {region_key(r.chrom, r.start, r.end): genome[r.chrom][r.start:r.end]
                for r in regions.itertuples() if r.chrom in genome}


def _parse_fasta_text(lines: Iterable[str]) -> Dict[str, str]:
    seqs, key, buf = {}, None, []
    for line in lines:
        if line.startswith(">"):
            if key is not None:
                seqs[key] = "".join(buf)
            key, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    if key is not None:
        seqs[key] = "".join(buf)
    return seqs


def check_reference_naming(genome: Genome, regions_bed: Optional[Path]) -> None:
    """
    Fail early if the genome or the negatives-region BED don't use chr-prefixed names for
    the main chromosomes. Unplaced scaffolds (GL000..., KI270...) are ignored.
    """
    names = genome.chrom_names()
    if names and not any(n in VALID_CHROMS for n in names):
        if any(normalise_chrom(n) in VALID_CHROMS for n in names):
            raise PipelineError("Genome FASTA uses Ensembl-style chromosome names (1, 2, X...). "
                                "Use a GENCODE/UCSC GRCh38 FASTA with chr-prefixed names.")
        raise PipelineError(f"Genome FASTA has no recognised human chromosomes (found e.g. {names[:3]}).")

    if regions_bed is not None:
        main, ensembl = 0, 0
        with open(regions_bed) as f:
            for line in f:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                chrom = line.split("\t", 1)[0]
                if chrom in VALID_CHROMS:
                    main += 1
                    break
                if normalise_chrom(chrom) in VALID_CHROMS:
                    ensembl += 1
        if main == 0:
            if ensembl:
                raise PipelineError(f"{regions_bed} uses chromosome names without the 'chr' prefix; "
                                    "it must match the genome FASTA (chr1, chr2, ...).")
            raise PipelineError(f"{regions_bed} has no intervals on chr1-22/X/Y.")
