#!/usr/bin/env python3
"""
run_rbp_analysis.py
===================
Generalised RNA-binding protein (RBP) analysis pipeline.

Given an ENCODE-style eCLIP BED file (GRCh38, chr-prefixed), this script:
  1. Validates and cleans the BED file (optional ENCODE significance filters)
  2. Generates length- and chromosome-matched negative regions (bedtools shuffle)
  3. Extracts strand-aware sequences from the genome
  4. Trains CNN + LR models on that protein's own data (chromosome split)
  5. Evaluates on held-out test chromosomes -> AUC
  6. Saves ROC curve + score distribution plots
  7. Annotates peaks (exonic/intronic/UTR/intergenic)
  8. Generates predictions CSV (with split labels) + Word report

Usage:
    python run_rbp_analysis.py --input peaks.bed --name MY_PROTEIN \\
        --genome GRCh38.fa --genes-bed genes_only.bed [--gtf gencode.gtf]

Optional:
    --test-chroms chr21 chr22 chrX     (default)
    --val-chroms  chr19 chr20          (default)
    --negatives   path/to/neg.bed      (skip generation, use existing)
    --min-log2fc 3 --min-log10p 3      (ENCODE-recommended peak thresholds)
    --max-len 126                      (CNN input length; longer peaks centre-cropped)
    --skip-annotation / --skip-report

Outputs -> <output-dir>/<name>/
"""

import argparse
import json
import random
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# RUNTIME CONFIGURATION (set by configure() from command-line arguments)
# ─────────────────────────────────────────────────────────────────────────────
GENOME_FA: Optional[Path] = None
GENES_BED: Optional[Path] = None
GTF_FILE:  Optional[Path] = None

RANDOM_STATE = 42
MAX_LEN      = 126
MAX_N_FRACTION = 0.10      # sequences with more ambiguous bases than this are dropped
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR   = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generalised RBP analysis pipeline")
    p.add_argument("--input",    required=True, help="Path to eCLIP BED file")
    p.add_argument("--name",     required=True, help="Protein name (used for output folder/filenames)")
    p.add_argument("--genome",   required=True, help="Path to GRCh38 genome FASTA (chr-prefixed names)")
    p.add_argument("--gtf",      default=None, help="(Optional) GENCODE GTF for region annotation")
    p.add_argument("--genes-bed", default=None,
                   help="Gene intervals BED used to constrain negatives (required unless --negatives)")
    p.add_argument("--output-dir", default="./results", help="Output directory (default: ./results)")
    p.add_argument("--negatives", default=None, help="(Optional) pre-computed negatives BED — skips generation")
    p.add_argument("--test-chroms", nargs="+", default=["chr21", "chr22", "chrX"],
                   help="Chromosomes held out for final AUC test (default: chr21 chr22 chrX)")
    p.add_argument("--val-chroms",  nargs="+", default=["chr19", "chr20"],
                   help="Chromosomes held out for validation (default: chr19 chr20)")
    p.add_argument("--min-log2fc", type=float, default=None,
                   help="Keep peaks with log2 fold-enrichment (BED column 7) >= this. ENCODE uses 3.")
    p.add_argument("--min-log10p", type=float, default=None,
                   help="Keep peaks with -log10 p-value (BED column 8) >= this. ENCODE uses 3.")
    p.add_argument("--max-len", type=int, default=126,
                   help="CNN input length in bp; longer sequences are centre-cropped (default: 126)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--skip-annotation", action="store_true", help="Skip GTF region annotation")
    p.add_argument("--skip-report", action="store_true", help="Skip Word report generation")
    return p.parse_args(argv)


def configure(args) -> None:
    """Set module-level paths/parameters from parsed arguments and check they exist."""
    global GENOME_FA, GENES_BED, GTF_FILE, MAX_LEN, RANDOM_STATE

    GENOME_FA = Path(args.genome)
    if not GENOME_FA.exists():
        sys.exit(f"ERROR: genome FASTA not found: {GENOME_FA}")

    if args.negatives is None and args.genes_bed is None:
        sys.exit("ERROR: --genes-bed is required unless --negatives is provided")
    GENES_BED = Path(args.genes_bed) if args.genes_bed else None
    if GENES_BED is not None and not GENES_BED.exists():
        sys.exit(f"ERROR: genes BED not found: {GENES_BED}")

    GTF_FILE = Path(args.gtf) if args.gtf else None
    if GTF_FILE is not None and not GTF_FILE.exists():
        sys.exit(f"ERROR: GTF not found: {GTF_FILE}")

    if args.negatives and not Path(args.negatives).exists():
        sys.exit(f"ERROR: negatives BED not found: {args.negatives}")

    if args.max_len < 8:
        sys.exit("ERROR: --max-len must be at least 8")
    MAX_LEN = args.max_len
    RANDOM_STATE = args.seed


def set_seed(seed: int) -> None:
    """Seed every random number generator the pipeline uses."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_output_dir(name: str, output_dir: str = "./results") -> Path:
    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CHROMOSOME NAMING & GENOME CHECKS
# ─────────────────────────────────────────────────────────────────────────────

VALID_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY", "chrM"}


def normalise_chrom(chrom: str) -> str:
    """Convert Ensembl-style names (1, X, MT) to UCSC/GENCODE style (chr1, chrX, chrM)."""
    if chrom.startswith("chr"):
        return chrom
    if chrom in ("MT", "M"):
        return "chrM"
    return "chr" + chrom


def ensure_fai() -> Optional[Path]:
    """Return the genome .fai index, creating it with samtools if possible."""
    fai = Path(str(GENOME_FA) + ".fai")
    if not fai.exists():
        try:
            subprocess.run(["samtools", "faidx", str(GENOME_FA)],
                           check=True, capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
    return fai if fai.exists() else None


def read_fai_sizes(fai: Path) -> Dict[str, int]:
    sizes = {}
    with open(fai) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                sizes[p[0]] = int(p[1])
    return sizes


def genome_chrom_names() -> List[str]:
    fai = ensure_fai()
    if fai is not None:
        return list(read_fai_sizes(fai))
    names = []
    with open(GENOME_FA) as f:
        for line in f:
            if line.startswith(">"):
                names.append(line[1:].split()[0])
    return names


def check_reference_naming() -> None:
    """Fail early if the genome or genes BED don't use chr-prefixed chromosome names."""
    names = genome_chrom_names()
    if names and not any(n.startswith("chr") for n in names):
        sys.exit("ERROR: genome FASTA uses Ensembl-style chromosome names (1, 2, X...). "
                 "Use a GENCODE/UCSC GRCh38 FASTA with chr-prefixed names.")
    if GENES_BED is not None:
        with open(GENES_BED) as f:
            for line in f:
                if line.strip() and not line.startswith(("#", "track", "browser")):
                    if not line.split("\t")[0].startswith("chr"):
                        sys.exit("ERROR: genes BED uses chromosome names without 'chr' prefix; "
                                 "it must match the genome FASTA naming.")
                    break


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — VALIDATE & CLEAN BED
# ─────────────────────────────────────────────────────────────────────────────

BED_COLS = ["chrom", "start", "end", "name", "score", "strand"]


def _to_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except ValueError:
        return None


def validate_bed(bed_path: Path, out_dir: Path,
                 min_log2fc: Optional[float] = None,
                 min_log10p: Optional[float] = None) -> pd.DataFrame:
    print("\n[1/8] Validating BED file...")
    rows = []
    skipped = renamed = filtered = 0
    with open(bed_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            cols = line.split("\t")
            if len(cols) < 3:
                skipped += 1; continue
            raw_chrom = cols[0]
            chrom = normalise_chrom(raw_chrom)
            try:
                start, end = int(cols[1]), int(cols[2])
            except ValueError:
                skipped += 1; continue
            if start < 0 or end <= start or chrom not in VALID_CHROMS:
                skipped += 1; continue
            strand = cols[5] if len(cols) > 5 else "."
            if strand not in ("+", "-", "."):
                skipped += 1; continue

            log2fc = _to_float(cols[6]) if len(cols) > 6 else None
            log10p = _to_float(cols[7]) if len(cols) > 7 else None
            if min_log2fc is not None and (log2fc is None or log2fc < min_log2fc):
                filtered += 1; continue
            if min_log10p is not None and (log10p is None or log10p < min_log10p):
                filtered += 1; continue

            if chrom != raw_chrom:
                renamed += 1
            rows.append({
                "chrom":  chrom,
                "start":  start,
                "end":    end,
                "name":   cols[3] if len(cols) > 3 else ".",
                "score":  _to_float(cols[4]) if len(cols) > 4 else 0.0,
                "strand": strand,
                "log2fc": log2fc,
                "log10p": log10p,
            })

    if not rows:
        sys.exit(f"ERROR: no usable peaks in {bed_path} "
                 f"({skipped:,} invalid lines, {filtered:,} removed by significance filters). "
                 "Expected tab-separated BED with chromosomes chr1-22/X/Y/M (or 1-22/X/Y/MT).")

    df = pd.DataFrame(rows)
    before = len(df)
    # Same interval on opposite strands = two different RNA binding events, so keep strand in the key
    df = df.drop_duplicates(subset=["chrom", "start", "end", "strand"]).reset_index(drop=True)
    df["peak_length"] = df["end"] - df["start"]
    df["peak_id"] = [f"peak_{i}" for i in range(len(df))]

    print(f"  Peaks loaded        : {len(df):,}")
    print(f"  Skipped (invalid)   : {skipped:,}")
    if min_log2fc is not None or min_log10p is not None:
        print(f"  Removed by filters  : {filtered:,} (log2FC >= {min_log2fc}, -log10p >= {min_log10p})")
    if renamed:
        print(f"  Chrom names fixed   : {renamed:,} (e.g. '1' -> 'chr1')")
    print(f"  Duplicates removed  : {before - len(df):,}")
    print(f"  Chromosomes         : {sorted(df['chrom'].unique())}")
    print(f"  Peak length range   : {df['peak_length'].min()}–{df['peak_length'].max()} bp")
    print(f"  Median peak length  : {int(df['peak_length'].median())} bp")
    n_unstranded = int((df["strand"] == ".").sum())
    if n_unstranded:
        print(f"  WARNING: {n_unstranded:,} peaks have no strand ('.'); the + strand will be used. "
              "eCLIP is stranded — check your input.")

    cleaned = out_dir / "peaks_cleaned.bed"
    out = df[["chrom", "start", "end", "peak_id", "score", "strand"]].copy()
    out["score"] = out["score"].fillna(0)
    out.to_csv(cleaned, sep="\t", index=False, header=False)
    print(f"  Cleaned BED saved   : {cleaned}")
    return df


def read_bed(path: Path, id_prefix: str) -> pd.DataFrame:
    """Read a simple BED (>=3 columns) into a DataFrame with a stable ID per row."""
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3:
                continue
            try:
                start, end = int(cols[1]), int(cols[2])
            except ValueError:
                continue
            chrom = normalise_chrom(cols[0])
            if end <= start or chrom not in VALID_CHROMS:
                continue
            rows.append({"chrom": chrom, "start": start, "end": end,
                         "strand": cols[5] if len(cols) > 5 and cols[5] in ("+", "-") else "."})
    df = pd.DataFrame(rows, columns=["chrom", "start", "end", "strand"])
    df[f"{id_prefix}_id"] = [f"{id_prefix}_{i}" for i in range(len(df))]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — GENERATE MATCHED NEGATIVES
# ─────────────────────────────────────────────────────────────────────────────

GRCH38_SIZES = {
    "chr1":248956422,"chr2":242193529,"chr3":198295559,"chr4":190214555,
    "chr5":181538259,"chr6":170805979,"chr7":159345973,"chr8":145138636,
    "chr9":138394717,"chr10":133797422,"chr11":135086622,"chr12":133275309,
    "chr13":114364328,"chr14":107043718,"chr15":101991189,"chr16":90338345,
    "chr17":83257441,"chr18":80373285,"chr19":58617616,"chr20":64444167,
    "chr21":46709983,"chr22":50818468,"chrX":156040895,"chrY":57227415,
}


def generate_negatives(out_dir: Path, name: str) -> Path:
    print("\n[2/8] Generating matched negative regions...")

    peaks_bed   = out_dir / "peaks_cleaned.bed"
    neg_bed_out = out_dir / f"{name}_negatives.bed"
    genome_file = out_dir / "genome.txt"

    fai = ensure_fai()
    if fai is not None:
        chrom_sizes = {c: s for c, s in read_fai_sizes(fai).items() if c in VALID_CHROMS}
    else:
        print("  Note: genome .fai unavailable — using standard GRCh38 chromosome sizes")
        chrom_sizes = GRCH38_SIZES

    # Each shuffled region keeps its peak's length (shuffle always does) and its peak's
    # chromosome, so every split gets negatives from the same chromosomes as its positives.
    # bedtools' own -chrom flag is silently ignored when combined with -incl (v2.31), so we
    # shuffle one chromosome at a time instead.
    peak_lines: Dict[str, List[str]] = {}
    with open(peaks_bed) as f:
        for line in f:
            peak_lines.setdefault(line.split("\t", 1)[0], []).append(line)
    gene_lines: Dict[str, List[str]] = {}
    with open(GENES_BED) as f:
        for line in f:
            if line.strip() and not line.startswith(("#", "track", "browser")):
                gene_lines.setdefault(line.split("\t", 1)[0], []).append(line)

    chrom_peaks = out_dir / "_shuffle_peaks.bed"
    chrom_genes = out_dir / "_shuffle_genes.bed"
    outputs, skipped = [], []
    try:
        for i, chrom in enumerate(sorted(peak_lines)):
            if chrom not in chrom_sizes or chrom not in gene_lines:
                skipped.append(chrom)
                continue
            genome_file.write_text(f"{chrom}\t{chrom_sizes[chrom]}\n")
            chrom_peaks.write_text("".join(peak_lines[chrom]))
            chrom_genes.write_text("".join(gene_lines[chrom]))
            cmd = [
                "bedtools", "shuffle",
                "-i", str(chrom_peaks),
                "-g", str(genome_file),
                "-incl", str(chrom_genes),
                "-excl", str(peaks_bed),
                "-noOverlapping",
                "-maxTries", "1000",
                "-seed", str(RANDOM_STATE + i),
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                sys.exit(f"ERROR: bedtools shuffle failed on {chrom}:\n{e.stderr}")
            except FileNotFoundError:
                sys.exit("ERROR: bedtools not found. Install with: sudo apt install bedtools")
            outputs.append(result.stdout)
    finally:
        chrom_peaks.unlink(missing_ok=True)
        chrom_genes.unlink(missing_ok=True)

    if skipped:
        print(f"  WARNING: no gene intervals / size for {skipped}; no negatives generated there")
    with open(neg_bed_out, "w") as f:
        f.write("".join(outputs))
    n_neg = sum(len([l for l in o.splitlines() if l.strip()]) for o in outputs)
    print(f"  Negatives generated : {n_neg:,} (length- and chromosome-matched)")
    return neg_bed_out


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — EXTRACT STRAND-AWARE SEQUENCES
# ─────────────────────────────────────────────────────────────────────────────

_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def clean_seq(seq: str) -> str:
    """Uppercase, U->T, and replace any non-ACGT character with N (keeps positions intact)."""
    return re.sub(r"[^ACGTN]", "N", seq.upper().replace("U", "T"))


def revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def region_key(chrom: str, start: int, end: int) -> str:
    """samtools-style 1-based inclusive region for a 0-based half-open BED interval."""
    return f"{chrom}:{start + 1}-{end}"


def fetch_via_samtools(regions: pd.DataFrame, tmp_dir: Path) -> Optional[Dict[str, str]]:
    """Fetch + strand sequences for each region. Returns {region_key: seq} or None if unavailable."""
    fai = ensure_fai()
    if fai is None:
        return None
    available = set(read_fai_sizes(fai))
    keys = sorted({region_key(r.chrom, r.start, r.end)
                   for r in regions.itertuples() if r.chrom in available})
    tmp = tmp_dir / "_regions_tmp.txt"
    tmp.write_text("\n".join(keys) + "\n")
    try:
        result = subprocess.run(["samtools", "faidx", str(GENOME_FA), "-r", str(tmp)],
                                capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    finally:
        tmp.unlink(missing_ok=True)

    seqs, key, buf = {}, None, []
    for line in result.stdout.splitlines():
        if line.startswith(">"):
            if key is not None:
                seqs[key] = "".join(buf)
            key, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    if key is not None:
        seqs[key] = "".join(buf)
    return seqs


def fetch_in_memory(regions: pd.DataFrame) -> Dict[str, str]:
    print("  Loading genome into memory (~2-3 min for GRCh38)...")
    wanted = set(regions["chrom"])
    genome, hdr, buf = {}, None, []
    with open(GENOME_FA) as f:
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


def attach_sequences(df: pd.DataFrame, fetched: Dict[str, str],
                     id_col: str, out_fasta: Path) -> pd.DataFrame:
    """
    Add a 'sequence' column (RNA-sense: reverse-complemented for '-' strand),
    keyed by coordinates so rows can never be misaligned. Drops rows whose
    sequence could not be fetched or is mostly ambiguous bases.
    """
    seqs = []
    for r in df.itertuples():
        s = clean_seq(fetched.get(region_key(r.chrom, r.start, r.end), ""))
        if r.strand == "-":
            s = revcomp(s)
        seqs.append(s)
    out = df.copy()
    out["sequence"] = seqs
    n_frac = out["sequence"].map(lambda s: s.count("N") / len(s) if s else 1.0)
    keep = n_frac <= MAX_N_FRACTION
    dropped = int((~keep).sum())
    if dropped:
        print(f"  Dropped {dropped:,} regions (missing from genome or >{MAX_N_FRACTION:.0%} N)")
    out = out[keep].reset_index(drop=True)

    with open(out_fasta, "w") as f:
        for r in out.itertuples():
            f.write(f">{getattr(r, id_col)}::{region_key(r.chrom, r.start, r.end)}({r.strand})\n"
                    f"{r.sequence}\n")
    return out


def extract_all_sequences(pos_df: pd.DataFrame, neg_df: pd.DataFrame,
                          out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("\n[3/8] Extracting strand-aware sequences...")
    regions = pd.concat([pos_df[["chrom", "start", "end"]], neg_df[["chrom", "start", "end"]]])

    fetched = fetch_via_samtools(regions, out_dir)
    if fetched is not None:
        print("  Extracted via samtools (fast)")
    else:
        print("  samtools unavailable — falling back to in-memory genome")
        fetched = fetch_in_memory(regions)

    pos_df = attach_sequences(pos_df, fetched, "peak_id", out_dir / "positives.fa")
    neg_df = attach_sequences(neg_df, fetched, "neg_id",  out_dir / "negatives.fa")

    print(f"  Positive sequences  : {len(pos_df):,}")
    print(f"  Negative sequences  : {len(neg_df):,}")
    if len(pos_df):
        lens = pos_df["sequence"].str.len()
        print(f"  Length range        : {lens.min()}–{lens.max()} bp  (median {int(lens.median())} bp)")
        n_long = int((lens > MAX_LEN).sum())
        if n_long:
            print(f"  Note: {n_long:,} positives longer than --max-len {MAX_LEN} bp "
                  "will be centre-cropped for the CNN")
    if len(pos_df) == 0 or len(neg_df) == 0:
        sys.exit("ERROR: no sequences extracted — check that the genome FASTA matches your BED (GRCh38, chr names).")
    return pos_df, neg_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — BUILD TRAIN / VAL / TEST SPLITS
# ─────────────────────────────────────────────────────────────────────────────

def assign_split(chrom: str, test_set: set, val_set: set) -> str:
    if chrom in test_set:
        return "test"
    if chrom in val_set:
        return "val"
    return "train"


def build_splits(pos_df: pd.DataFrame, neg_df: pd.DataFrame,
                 test_chroms: List[str], val_chroms: List[str]):
    """
    Split positives and negatives by chromosome, then balance each split by downsampling.
    Returns ({split: (seqs, labels)}, pos_df with 'split', neg_df with 'split').
    """
    test_set, val_set = set(test_chroms), set(val_chroms)
    if test_set & val_set:
        sys.exit(f"ERROR: chromosomes in both test and val sets: {sorted(test_set & val_set)}")

    present = set(pos_df["chrom"])
    for label, chroms in (("test", test_set), ("val", val_set)):
        missing = sorted(chroms - present)
        if missing:
            print(f"  Note: {label} chromosomes with no peaks: {missing}")

    pos_df = pos_df.copy()
    neg_df = neg_df.copy()
    pos_df["split"] = [assign_split(c, test_set, val_set) for c in pos_df["chrom"]]
    neg_df["split"] = [assign_split(c, test_set, val_set) for c in neg_df["chrom"]]

    rng = random.Random(RANDOM_STATE)
    data = {}
    for split in ("train", "val", "test"):
        pos = pos_df.loc[pos_df["split"] == split, "sequence"].tolist()
        neg = neg_df.loc[neg_df["split"] == split, "sequence"].tolist()
        n = min(len(pos), len(neg))
        rng.shuffle(pos); rng.shuffle(neg)
        combined = [(s, 1) for s in pos[:n]] + [(s, 0) for s in neg[:n]]
        rng.shuffle(combined)
        data[split] = ([s for s, _ in combined], [l for _, l in combined])
        print(f"  {split.capitalize():<5} set           : {2*n:,} sequences ({n:,} pos / {n:,} neg)")

    for split, hint in (("test", "--test-chroms"), ("val", "--val-chroms"), ("train", "the split")):
        if len(data[split][0]) == 0:
            sys.exit(f"ERROR: {split} set is empty. Chromosomes in data: {sorted(present)}. "
                     f"Adjust {hint} to chromosomes present in your data.")
    return data, pos_df, neg_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — CNN MODEL
# ─────────────────────────────────────────────────────────────────────────────

class RBP_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64,  kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn2   = nn.BatchNorm1d(128)
        self.pool  = nn.MaxPool1d(kernel_size=4)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.max(x, dim=2).values
        return self.classifier(x).squeeze(1)


def fit_length(seq: str, length: int) -> str:
    """Centre-crop sequences longer than `length` (binding is usually near the peak centre)."""
    if len(seq) <= length:
        return seq
    offset = (len(seq) - length) // 2
    return seq[offset:offset + length]


def one_hot(seq: str) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}   # N stays all-zero
    arr = np.zeros((4, MAX_LEN), dtype=np.float32)
    for i, b in enumerate(fit_length(seq, MAX_LEN)):
        if b in mapping:
            arr[mapping[b], i] = 1.0
    return arr


class SeqDataset(Dataset):
    def __init__(self, seqs, labels=None):
        self.seqs   = seqs
        self.labels = labels
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        x = torch.tensor(one_hot(self.seqs[idx]))
        if self.labels is not None:
            return x, torch.tensor(self.labels[idx], dtype=torch.float32)
        return x


def train_cnn(train_seqs, train_labels, val_seqs, val_labels,
              out_dir: Path, name: str, max_epochs: int = 30) -> RBP_CNN:
    print("  Training CNN...")
    set_seed(RANDOM_STATE)
    model = RBP_CNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    loader_gen   = torch.Generator().manual_seed(RANDOM_STATE)
    train_loader = DataLoader(SeqDataset(train_seqs, train_labels), batch_size=64,
                              shuffle=True, generator=loader_gen)
    val_loader   = DataLoader(SeqDataset(val_seqs, val_labels), batch_size=64, shuffle=False)

    best_val_loss = float("inf")
    best_weights  = None
    patience, patience_count = 5, 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss += criterion(model(x), y).item()

        train_loss /= len(train_loader)
        val_loss   /= max(len(val_loader), 1)

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:2d} — train loss: {train_loss:.4f}  val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    if best_weights:
        model.load_state_dict(best_weights)

    weights_path = out_dir / f"{name}_cnn.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"  CNN weights saved   : {weights_path}")
    return model


def cnn_predict(model: RBP_CNN, seqs: List[str]) -> List[float]:
    loader = DataLoader(SeqDataset(seqs), batch_size=64, shuffle=False)
    model.eval()
    probs = []
    with torch.no_grad():
        for x in loader:
            probs.extend(torch.sigmoid(model(x.to(DEVICE))).cpu().numpy().tolist())
    return probs


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5b — LR MODEL
# ─────────────────────────────────────────────────────────────────────────────

def seq_to_kmers(seq: str, ks=(2, 3)) -> str:
    seq = clean_seq(seq)
    tokens = []
    for k in ks:
        tokens.extend(kmer for kmer in (seq[i:i+k] for i in range(len(seq) - k + 1))
                      if "N" not in kmer)
    return " ".join(tokens)


def train_lr(train_seqs, train_labels):
    print("  Training LR...")
    texts      = [seq_to_kmers(s) for s in train_seqs]
    vectorizer = CountVectorizer()
    X_train    = vectorizer.fit_transform(texts)
    clf        = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train, np.array(train_labels))
    print(f"  LR trained on {len(train_seqs):,} sequences")
    return clf, vectorizer


def lr_predict(clf, vectorizer, seqs: List[str]) -> List[float]:
    X = vectorizer.transform([seq_to_kmers(s) for s in seqs])
    return clf.predict_proba(X)[:, 1].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 (combined) — TRAIN & EVALUATE MODELS
# ─────────────────────────────────────────────────────────────────────────────

def train_and_score(data, out_dir, name):
    print("\n[4/8] Training and evaluating models...")

    train_seqs, train_labels = data["train"]
    val_seqs,   val_labels   = data["val"]
    test_seqs,  test_labels  = data["test"]

    cnn       = train_cnn(train_seqs, train_labels, val_seqs, val_labels, out_dir, name)
    cnn_probs = cnn_predict(cnn, test_seqs)
    cnn_auc   = roc_auc_score(test_labels, cnn_probs)
    print(f"  CNN  test AUC       : {cnn_auc:.4f}")

    clf, vec  = train_lr(train_seqs, train_labels)
    lr_probs  = lr_predict(clf, vec, test_seqs)
    lr_auc    = roc_auc_score(test_labels, lr_probs)
    print(f"  LR   test AUC       : {lr_auc:.4f}")

    mean_probs = [(c + l) / 2 for c, l in zip(cnn_probs, lr_probs)]
    mean_auc   = roc_auc_score(test_labels, mean_probs)
    print(f"  Ensemble test AUC   : {mean_auc:.4f}")

    return {
        "cnn_auc":    cnn_auc,
        "lr_auc":     lr_auc,
        "mean_auc":   mean_auc,
        "cnn_probs":  cnn_probs,
        "lr_probs":   lr_probs,
        "mean_probs": mean_probs,
        "test_labels": test_labels,
        "test_seqs":   test_seqs,
        "cnn_model":  cnn,
        "lr_clf":     clf,
        "lr_vec":     vec,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(results: dict, out_dir: Path, name: str):
    print("\n[5/8] Generating plots...")

    y          = np.array(results["test_labels"])
    cnn_probs  = np.array(results["cnn_probs"])
    lr_probs   = np.array(results["lr_probs"])
    mean_probs = np.array(results["mean_probs"])

    fig, ax = plt.subplots(figsize=(7, 6))
    for probs, auc, label, color in [
        (cnn_probs,  results["cnn_auc"],  "CNN",      "#2196F3"),
        (lr_probs,   results["lr_auc"],   "LR",       "#4CAF50"),
        (mean_probs, results["mean_auc"], "Ensemble", "#FF5722"),
    ]:
        fpr, tpr, _ = roc_curve(y, probs)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(f"ROC Curve (held-out test chromosomes) — {name}", fontsize=12)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = out_dir / "figures" / "roc_curve.png"
    plt.savefig(roc_path, dpi=150); plt.close()
    print(f"  ROC curve           : {roc_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, probs, model_name in [
        (axes[0], cnn_probs, "CNN"),
        (axes[1], lr_probs,  "LR"),
    ]:
        ax.hist(probs[y==1], bins=40, alpha=0.6, color="#2196F3",
                label="Positive (peaks)",    density=True)
        ax.hist(probs[y==0], bins=40, alpha=0.6, color="#F44336",
                label="Negative (shuffled)", density=True)
        ax.set_xlabel(f"{model_name} binding probability", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{model_name} test-set scores — {name}", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    dist_path = out_dir / "figures" / "score_distributions.png"
    plt.savefig(dist_path, dpi=150); plt.close()
    print(f"  Score distributions : {dist_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — ANNOTATE REGIONS
# ─────────────────────────────────────────────────────────────────────────────

def parse_gtf(gtf_path: Path) -> Dict:
    print("  Parsing GENCODE GTF (~1-2 min)...")
    features = {}
    target   = {"exon", "UTR", "CDS", "gene"}
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"): continue
            cols = line.strip().split("\t")
            if len(cols) < 9 or cols[2] not in target: continue
            chrom = cols[0]
            start = int(cols[3]) - 1
            end   = int(cols[4])
            m     = re.search(r'gene_name "([^"]+)"', cols[8])
            gene  = m.group(1) if m else "."
            features.setdefault(chrom, []).append((start, end, cols[2], gene))
    return features


def annotate_peak(chrom, start, end, gtf):
    if chrom not in gtf:
        return "intergenic", "."
    overlaps = [f for f in gtf[chrom] if f[1] > start and f[0] < end]
    if not overlaps:
        return "intergenic", "."
    gene  = overlaps[0][3]
    types = {f[2] for f in overlaps}
    if "UTR"  in types: return "UTR",      gene
    if "exon" in types or "CDS" in types: return "exonic", gene
    return "intronic", gene


def annotate_regions(peaks_df: pd.DataFrame, skip: bool) -> pd.DataFrame:
    print("\n[6/8] Annotating genomic regions...")
    peaks_df = peaks_df.copy()
    if skip or GTF_FILE is None:
        print("  Skipped (--skip-annotation)" if skip else "  Skipped (no --gtf given)")
        peaks_df["region_type"] = "not_annotated"
        peaks_df["gene_name"]   = "."
        return peaks_df

    gtf = parse_gtf(GTF_FILE)
    annotations = [annotate_peak(r.chrom, r.start, r.end, gtf) for r in peaks_df.itertuples()]
    peaks_df["region_type"] = [a[0] for a in annotations]
    peaks_df["gene_name"]   = [a[1] for a in annotations]

    counts = peaks_df["region_type"].value_counts()
    print("  Region breakdown:")
    for rt, cnt in counts.items():
        print(f"    {rt:<15} : {cnt:>5,}  ({100*cnt/len(peaks_df):.1f}%)")
    return peaks_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — SAVE CSV + WORD REPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(pos_df: pd.DataFrame, results: dict, out_dir: Path, name: str) -> pd.DataFrame:
    """
    Score every positive peak and record which split it came from. Only peaks with
    split == 'test' were unseen during training; train/val scores are optimistic.
    """
    print("\n[7/8] Saving predictions CSV...")
    df = pos_df.copy()
    seqs = df["sequence"].tolist()
    df["seq_length"] = [len(s) for s in seqs]
    df["cnn_prob"]   = cnn_predict(results["cnn_model"], seqs)
    df["lr_prob"]    = lr_predict(results["lr_clf"], results["lr_vec"], seqs)
    df["mean_prob"]  = (df["cnn_prob"] + df["lr_prob"]) / 2
    df["seen_in_training"] = df["split"] != "test"   # val is used for early stopping
    df["high_confidence"]  = (df["cnn_prob"] > 0.5) & (df["lr_prob"] > 0.5)
    df = df.sort_values("mean_prob", ascending=False).reset_index(drop=True)

    first = ["peak_id", "chrom", "start", "end", "strand", "name", "split", "seen_in_training",
             "cnn_prob", "lr_prob", "mean_prob", "high_confidence"]
    cols = [c for c in first if c in df.columns] + \
           [c for c in df.columns if c not in first and c != "sequence"] + ["sequence"]
    df = df[cols]

    csv_path = out_dir / f"{name}_predictions.csv"
    df.to_csv(csv_path, index=False)

    test = df[df["split"] == "test"]
    n_high = int(test["high_confidence"].sum())
    print(f"  CSV saved           : {csv_path}")
    print(f"  High-confidence     : {n_high:,} / {len(test):,} held-out test peaks "
          f"({100*n_high/max(len(test),1):.1f}%)")
    print("  Note: scores for train/val peaks (seen_in_training=True) are optimistic.")
    return df


def interpret_auc(auc: float) -> str:
    if auc >= 0.85:
        return f"Strong model performance (AUC = {auc:.3f}). The model reliably distinguishes binding from non-binding sites, indicating clear sequence-level features driving binding."
    elif auc >= 0.70:
        return f"Good model performance (AUC = {auc:.3f}). The model captures meaningful binding sequence features, though some binding sites may lack strong motif signal."
    elif auc >= 0.55:
        return f"Moderate model performance (AUC = {auc:.3f}). The model performs above chance but binding may be driven by context beyond short sequence motifs."
    else:
        return f"Low model performance (AUC = {auc:.3f}). Binding sites are not well-distinguished from matched negatives by sequence features alone."


def generate_report(peaks_df, results, out_dir, name, test_chroms, val_chroms):
    print("\n[8/8] Generating Word report...")
    js_str = json.dumps   # safely quote any text injected into the JavaScript

    cnn_auc  = results["cnn_auc"]
    lr_auc   = results["lr_auc"]
    mean_auc = results["mean_auc"]
    test_df  = peaks_df[peaks_df["split"] == "test"]
    n_high   = int(test_df["high_confidence"].sum())
    pct_high = 100 * n_high / len(test_df) if len(test_df) else 0
    interp   = interpret_auc(mean_auc)
    region_counts = peaks_df["region_type"].value_counts().to_dict()

    top5_js = "".join(
        f"""new TableRow({{ children: [
          cell({js_str(f"{r.chrom}:{r.start}-{r.end}({r.strand})")}, 2800),
          cell({js_str(str(r.gene_name))}, 1600),
          cell({js_str(str(r.region_type))}, 1400),
          cell({js_str(f"{r.cnn_prob:.4f}")}, 1000),
          cell({js_str(f"{r.lr_prob:.4f}")}, 1000),
          cell({js_str(f"{r.mean_prob:.4f}")}, 1000)
        ] }}),"""
        for r in test_df.head(5).itertuples()
    )
    region_rows = "".join(
        f"""new TableRow({{ children: [
          cell({js_str(str(rt))}, 3000), cell({js_str(str(cnt))}, 1500),
          cell({js_str(f"{100*cnt/len(peaks_df):.1f}%")}, 1500)
        ] }}),"""
        for rt, cnt in region_counts.items()
    )

    def row(label, value):
        return f"new TableRow({{ children: [cell({js_str(label)},3600), cell({js_str(value)},5760)] }}),"

    summary_rows = "".join([
        row("Protein", name),
        row("Peaks with sequences", f"{len(peaks_df):,}"),
        row("Held-out test peaks", f"{len(test_df):,}"),
        row("High-confidence test peaks", f"{n_high:,} ({pct_high:.1f}%)"),
        row("Test chromosomes", " ".join(test_chroms)),
        row("Val chromosomes", " ".join(val_chroms)),
        row("CNN AUC", f"{cnn_auc:.4f}"),
        row("LR AUC", f"{lr_auc:.4f}"),
        row("Ensemble AUC", f"{mean_auc:.4f}"),
    ])

    docx_module = SCRIPT_DIR / "node_modules" / "docx"
    report_path = out_dir / f"{name}_report.docx"
    methods = {
        "input": f"eCLIP BED file for {name}; {len(peaks_df):,} peaks with usable sequence on GRCh38.",
        "neg": ("Negative regions generated with bedtools shuffle, matched to each peak's length and "
                f"chromosome, constrained to genic regions and excluding peak coordinates. Seed = {RANDOM_STATE}."),
        "seq": ("Sequences extracted in the RNA sense: minus-strand peaks are reverse-complemented. "
                f"Sequences with more than {MAX_N_FRACTION:.0%} ambiguous bases were removed."),
        "split": (f"Chromosome-based split: validation ({' '.join(val_chroms)}), test ({' '.join(test_chroms)}), "
                  "training on all remaining chromosomes. Positives and negatives balanced by downsampling within each split."),
        "cnn": ("Conv1D(4->64, k=7) -> BN -> MaxPool(4) -> Conv1D(64->128, k=7) -> BN -> MaxPool(4) -> GlobalMaxPool "
                "-> FC(128->64, ReLU, Dropout 0.3) -> FC(64->1). Adam (lr=1e-3), BCEWithLogitsLoss, early stopping on "
                f"validation loss (patience=5), max 30 epochs. One-hot input of {MAX_LEN} bp (longer sequences centre-cropped)."),
        "lr": "k-mer logistic regression with 2-mer and 3-mer counts (CountVectorizer fit on training data only). Max iterations = 1000.",
        "eval": ("AUC computed on held-out test chromosomes only. Peak tables in this report show test-chromosome "
                 "peaks only, because scores for training peaks are optimistic."),
    }

    js = f"""
const {{ Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
         AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType }} = require({js_str(str(docx_module))});
const fs = require('fs');

const border  = {{ style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" }};
const borders = {{ top: border, bottom: border, left: border, right: border }};

function cell(text, w, bold=false, bg="FFFFFF") {{
  return new TableCell({{
    borders,
    width: {{ size: w, type: WidthType.DXA }},
    shading: {{ fill: bg, type: ShadingType.CLEAR }},
    margins: {{ top: 80, bottom: 80, left: 120, right: 120 }},
    children: [new Paragraph({{ children: [new TextRun({{ text: String(text), bold, size: 20 }})] }})]
  }});
}}
function hcell(text, w) {{ return cell(text, w, true, "D5E8F0"); }}
function h1(text) {{
  return new Paragraph({{ heading: HeadingLevel.HEADING_1,
    children: [new TextRun({{ text, bold: true, size: 32, font: "Arial" }})] }});
}}
function h2(text) {{
  return new Paragraph({{ heading: HeadingLevel.HEADING_2,
    children: [new TextRun({{ text, bold: true, size: 26, font: "Arial" }})] }});
}}
function para(text) {{
  return new Paragraph({{ children: [new TextRun({{ text, size: 22, font: "Arial" }})] }});
}}
function spacer() {{ return new Paragraph({{ children: [new TextRun("")] }}); }}

const doc = new Document({{
  styles: {{
    default: {{ document: {{ run: {{ font: "Arial", size: 22 }} }} }},
    paragraphStyles: [
      {{ id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
         run: {{ size: 32, bold: true, font: "Arial" }},
         paragraph: {{ spacing: {{ before: 240, after: 120 }}, outlineLevel: 0 }} }},
      {{ id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
         run: {{ size: 26, bold: true, font: "Arial" }},
         paragraph: {{ spacing: {{ before: 180, after: 80 }}, outlineLevel: 1 }} }},
    ]
  }},
  sections: [{{
    properties: {{ page: {{
      size: {{ width: 12240, height: 15840 }},
      margin: {{ top: 1440, right: 1440, bottom: 1440, left: 1440 }}
    }} }},
    children: [
      new Paragraph({{ alignment: AlignmentType.CENTER, children: [
        new TextRun({{ text: {js_str("RBP Binding Analysis: " + name)}, bold: true, size: 40, font: "Arial" }})
      ] }}),
      new Paragraph({{ alignment: AlignmentType.CENTER, children: [
        new TextRun({{ text: "Sequence-Based Binding Site Analysis Pipeline", size: 26, color: "555555", font: "Arial" }})
      ] }}),
      spacer(),

      h1("1. Summary"),
      new Table({{
        width: {{ size: 9360, type: WidthType.DXA }},
        columnWidths: [3600, 5760],
        rows: [
          new TableRow({{ children: [hcell("Parameter",3600), hcell("Value",5760)] }}),
          {summary_rows}
        ]
      }}),
      spacer(),

      h1("2. Model Performance"),
      para({js_str(interp)}),
      spacer(),
      para("See figures/roc_curve.png for the ROC curve and figures/score_distributions.png for score distributions."),
      spacer(),

      h1("3. Top 5 Held-Out Test Peaks by Binding Probability"),
      new Table({{
        width: {{ size: 9360, type: WidthType.DXA }},
        columnWidths: [2800, 1600, 1400, 1000, 1000, 1000],
        rows: [
          new TableRow({{ children: [
            hcell("Coordinates",2800), hcell("Gene",1600), hcell("Region",1400),
            hcell("CNN",1000), hcell("LR",1000), hcell("Mean",1000)
          ] }}),
          {top5_js}
        ]
      }}),
      spacer(),

      h1("4. Genomic Region Annotation"),
      new Table({{
        width: {{ size: 6000, type: WidthType.DXA }},
        columnWidths: [3000, 1500, 1500],
        rows: [
          new TableRow({{ children: [hcell("Region",3000), hcell("Count",1500), hcell("%",1500)] }}),
          {region_rows}
        ]
      }}),
      spacer(),

      h1("5. Methods"),
      h2("5.1 Input Data"), para({js_str(methods["input"])}), spacer(),
      h2("5.2 Negative Region Generation"), para({js_str(methods["neg"])}), spacer(),
      h2("5.3 Sequence Extraction"), para({js_str(methods["seq"])}), spacer(),
      h2("5.4 Train/Val/Test Split"), para({js_str(methods["split"])}), spacer(),
      h2("5.5 CNN"), para({js_str(methods["cnn"])}), spacer(),
      h2("5.6 Logistic Regression"), para({js_str(methods["lr"])}), spacer(),
      h2("5.7 Evaluation"), para({js_str(methods["eval"])}),
    ]
  }}]
}});

Packer.toBuffer(doc).then(buf => {{
  fs.writeFileSync({js_str(str(report_path))}, buf);
  console.log('Report saved.');
}}).catch(err => {{ console.error(err); process.exit(1); }});
"""

    try:
        check = subprocess.run(["node", "-e", f"require({js_str(str(docx_module))})"],
                               capture_output=True)
        if check.returncode != 0:
            print("  Installing docx npm package...")
            subprocess.run(["npm", "install", "--prefix", str(SCRIPT_DIR), "docx"],
                           check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"  Warning: could not set up Node.js 'docx' package ({e}); skipping Word report.")
        print("  CSV and figures were still saved.")
        return

    js_path = out_dir / "_report.js"
    js_path.write_text(js)
    result = subprocess.run(["node", str(js_path)], capture_output=True, text=True)
    js_path.unlink(missing_ok=True)

    if result.returncode == 0:
        print(f"  Word report saved   : {report_path}")
    else:
        print(f"  Warning: Word report failed: {result.stderr[:300]}")
        print("  CSV and figures were still saved.")


def save_run_info(args, out_dir: Path) -> None:
    """Record the exact parameters and software versions used for this run."""
    import sklearn
    info = {
        "arguments": vars(args),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "max_len": MAX_LEN,
        "seed": RANDOM_STATE,
        "device": str(DEVICE),
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    configure(args)
    set_seed(RANDOM_STATE)
    name = args.name
    bed_in = Path(args.input)

    print("=" * 65)
    print(f"  RBP ANALYSIS PIPELINE")
    print(f"  Protein      : {name}")
    print(f"  Input        : {bed_in}")
    print(f"  Genome       : {GENOME_FA}")
    print(f"  Test chroms  : {args.test_chroms}")
    print(f"  Val chroms   : {args.val_chroms}")
    print(f"  Device       : {DEVICE}")
    print("=" * 65)

    if not bed_in.exists():
        sys.exit(f"ERROR: Input file not found: {bed_in}")

    out_dir = setup_output_dir(name, args.output_dir)
    print(f"  Output dir   : {out_dir}")
    save_run_info(args, out_dir)
    check_reference_naming()

    peaks_df = validate_bed(bed_in, out_dir, args.min_log2fc, args.min_log10p)

    if args.negatives:
        neg_bed = Path(args.negatives)
        print(f"\n[2/8] Using provided negatives: {neg_bed}")
    else:
        neg_bed = generate_negatives(out_dir, name)
    neg_df = read_bed(neg_bed, "neg")

    pos_df, neg_df = extract_all_sequences(peaks_df, neg_df, out_dir)

    print("\n[3/8] Building chromosome splits...")
    data, pos_df, neg_df = build_splits(pos_df, neg_df, args.test_chroms, args.val_chroms)

    results = train_and_score(data, out_dir, name)
    make_plots(results, out_dir, name)

    pos_df = annotate_regions(pos_df, args.skip_annotation)
    pos_df = save_csv(pos_df, results, out_dir, name)

    if args.skip_report:
        print("\n[8/8] Word report skipped (--skip-report)")
    else:
        generate_report(pos_df, results, out_dir, name, args.test_chroms, args.val_chroms)

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65)
    print(f"  CNN  AUC : {results['cnn_auc']:.4f}")
    print(f"  LR   AUC : {results['lr_auc']:.4f}")
    print(f"  Mean AUC : {results['mean_auc']:.4f}")
    print(f"\n  Results  : {out_dir}")
    print("=" * 65)
    return results


if __name__ == "__main__":
    main()
