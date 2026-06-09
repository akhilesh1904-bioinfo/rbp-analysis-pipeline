#!/usr/bin/env python3
"""
run_rbp_analysis.py
===================
Generalised RNA-binding protein (RBP) analysis pipeline.

Given any eCLIP BED file, this script:
  1. Validates and cleans the BED file
  2. Generates matched negative regions via bedtools shuffle
  3. Extracts sequences from GRCh38
  4. Trains CNN + LR models on that protein's own data (chromosome split)
  5. Evaluates on held-out test chromosomes → AUC
  6. Saves ROC curve + score distribution plots
  7. Annotates peaks (exonic/intronic/UTR/intergenic)
  8. Generates predictions CSV + Word report

Usage:
    python run_rbp_analysis.py --input data/MY_PROTEIN_peaks.bed --name MY_PROTEIN

Optional:
    --test-chroms chr21 chr22 chrX     (default)
    --val-chroms  chr19 chr20           (default)
    --skip-annotation                   (skip GTF step, faster)
    --negatives   path/to/neg.bed       (skip generation, use existing)

Outputs → results/MY_PROTEIN/
"""

import argparse
import os
import random
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

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
# FIXED PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = Path("/mnt/c/Users/AKHILESH NAIK/Desktop/RBP_Internship")
GENOME_FA    = Path("/mnt/c/Users/AKHILESH NAIK/Desktop/MSc_TDP43_Project/Data/Raw/GRCh38.primary_assembly.genome.fa")
GENES_BED    = Path("/mnt/c/Users/AKHILESH NAIK/Desktop/MSc_TDP43_Project/Data/Raw/genes_only.bed")
GTF_FILE     = Path("/mnt/c/Users/AKHILESH NAIK/Desktop/MSc_TDP43_Project/Data/Raw/gencode.v49.primary_assembly.annotation.gtf")

RANDOM_STATE = 42
MAX_LEN      = 126
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generalised RBP analysis pipeline")
    p.add_argument("--input",    required=True, help="Path to eCLIP BED file")
    p.add_argument("--name",     required=True, help="Protein name (used for output folder/filenames)")
    p.add_argument("--genome", required=True, help="Path to GRCh38 genome FASTA")
    p.add_argument("--gtf", required=False, default=None, help="Path to GENCODE GTF file")
    p.add_argument("--genes-bed", required=True, help="Path to genes_only.bed")
    p.add_argument("--output-dir", default="./results", help="Output directory (default: ./results)")
    p.add_argument("--negatives", default=None, help="(Optional) pre-computed negatives BED — skips generation")
    p.add_argument("--test-chroms", nargs="+", default=["chr21", "chr22", "chrX"],
                   help="Chromosomes held out for final AUC test (default: chr21 chr22 chrX)")
    p.add_argument("--val-chroms",  nargs="+", default=["chr19", "chr20"],
                   help="Chromosomes held out for validation (default: chr19 chr20)")
    p.add_argument("--skip-annotation", action="store_true",
                   help="Skip GTF region annotation (faster)")
    return p.parse_args()


def setup_output_dir(name: str, output_dir: str = "./results") -> Path:
    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — VALIDATE & CLEAN BED
# ─────────────────────────────────────────────────────────────────────────────

VALID_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY", "chrM"}

def validate_bed(bed_path: Path, out_dir: Path) -> pd.DataFrame:
    print("\n[1/8] Validating BED file...")
    rows, skipped = [], 0
    with open(bed_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "track", "browser")):
                continue
            cols = line.split("\t")
            if len(cols) < 3:
                skipped += 1; continue
            chrom = cols[0]
            try:
                start, end = int(cols[1]), int(cols[2])
            except ValueError:
                skipped += 1; continue
            if end <= start or chrom not in VALID_CHROMS:
                skipped += 1; continue
            rows.append({
                "chrom":  chrom,
                "start":  start,
                "end":    end,
                "name":   cols[3] if len(cols) > 3 else ".",
                "score":  float(cols[4]) if len(cols) > 4 else 0.0,
                "strand": cols[5] if len(cols) > 5 else ".",
            })

    df = pd.DataFrame(rows)
    before = len(df)
    df = df.drop_duplicates(subset=["chrom", "start", "end"])
    df["peak_length"] = df["end"] - df["start"]

    print(f"  Peaks loaded        : {len(df):,}")
    print(f"  Skipped (invalid)   : {skipped:,}")
    print(f"  Duplicates removed  : {before - len(df):,}")
    print(f"  Chromosomes         : {sorted(df['chrom'].unique())}")
    print(f"  Peak length range   : {df['peak_length'].min()}–{df['peak_length'].max()} bp")
    print(f"  Median peak length  : {int(df['peak_length'].median())} bp")

    cleaned = out_dir / "peaks_cleaned.bed"
    df[["chrom","start","end","name","score","strand"]].to_csv(
        cleaned, sep="\t", index=False, header=False)
    print(f"  Cleaned BED saved   : {cleaned}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — GENERATE MATCHED NEGATIVES
# ─────────────────────────────────────────────────────────────────────────────

def generate_negatives(peaks_df: pd.DataFrame, out_dir: Path, name: str) -> Path:
    print("\n[2/8] Generating matched negative regions...")

    peaks_bed   = out_dir / "peaks_cleaned.bed"
    neg_bed_out = out_dir / f"{name}_negatives.bed"
    genome_file = out_dir / "genome.txt"

    # Build genome sizes from .fai index if available, else use GRCh38 defaults
    fai = Path(str(GENOME_FA) + ".fai")
    if fai.exists():
        chrom_sizes = {}
        with open(fai) as f:
            for line in f:
                p = line.strip().split("\t")
                if p[0] in VALID_CHROMS:
                    chrom_sizes[p[0]] = int(p[1])
    else:
        print("  Note: genome .fai not found — using standard GRCh38 chromosome sizes")
        chrom_sizes = {
            "chr1":248956422,"chr2":242193529,"chr3":198295559,"chr4":190214555,
            "chr5":181538259,"chr6":170805979,"chr7":159345973,"chr8":145138636,
            "chr9":138394717,"chr10":133797422,"chr11":135086622,"chr12":133275309,
            "chr13":114364328,"chr14":107043718,"chr15":101991189,"chr16":90338345,
            "chr17":83257441,"chr18":80373285,"chr19":58617616,"chr20":64444167,
            "chr21":46709983,"chr22":50818468,"chrX":156040895,"chrY":57227415,
        }

    with open(genome_file, "w") as f:
        for chrom in sorted(chrom_sizes):
            f.write(f"{chrom}\t{chrom_sizes[chrom]}\n")

    cmd = [
        "bedtools", "shuffle",
        "-i", str(peaks_bed),
        "-g", str(genome_file),
        "-incl", str(GENES_BED),
        "-excl", str(peaks_bed),
        "-noOverlapping",
        "-maxTries", "1000",
        "-seed", str(RANDOM_STATE),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        with open(neg_bed_out, "w") as f:
            f.write(result.stdout)
        n_neg = result.stdout.strip().count("\n") + 1 if result.stdout.strip() else 0
        print(f"  Negatives generated : {n_neg:,}")
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: bedtools failed:\n{e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("  ERROR: bedtools not found. Install with: sudo apt install bedtools")
        sys.exit(1)

    return neg_bed_out


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — EXTRACT SEQUENCES FROM GRCh38
# ─────────────────────────────────────────────────────────────────────────────

def clean_seq(seq: str) -> str:
    return re.sub(r"[^ACGT]", "", seq.upper().replace("U", "T"))


def extract_via_samtools(bed_path: Path, out_fasta: Path) -> bool:
    regions, tmp = [], out_fasta.parent / "_regions_tmp.txt"
    with open(bed_path) as f:
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) >= 3:
                regions.append(f"{cols[0]}:{int(cols[1])+1}-{cols[2]}")
    with open(tmp, "w") as f:
        f.write("\n".join(regions) + "\n")
    try:
        result = subprocess.run(
            ["samtools", "faidx", str(GENOME_FA), "-r", str(tmp)],
            capture_output=True, text=True, check=True)
        with open(out_fasta, "w") as f:
            f.write(result.stdout)
        tmp.unlink(missing_ok=True)
        return True
    except Exception:
        tmp.unlink(missing_ok=True)
        return False


def load_genome_into_memory() -> Dict[str, str]:
    print("  Loading genome into memory (~2-3 min)...")
    seqs, hdr, buf = {}, None, []
    with open(GENOME_FA) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if hdr: seqs[hdr] = "".join(buf)
                hdr, buf = line[1:].split()[0], []
            else:
                buf.append(line.upper())
    if hdr: seqs[hdr] = "".join(buf)
    return seqs


def extract_from_bed_with_genome(bed_path: Path, out_fasta: Path, genome: Dict):
    with open(out_fasta, "w") as f:
        with open(bed_path) as b:
            for line in b:
                cols = line.strip().split("\t")
                if len(cols) < 3: continue
                chrom, start, end = cols[0], int(cols[1]), int(cols[2])
                if chrom in genome:
                    seq = clean_seq(genome[chrom][start:end])
                    if seq:
                        f.write(f">{chrom}:{start}-{end}\n{seq}\n")


def read_fasta_with_coords(path: Path) -> Tuple[List[str], List[str]]:
    """Returns (sequences, chromosomes)."""
    seqs, chroms, buf, chrom = [], [], [], None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if buf and chrom:
                    seqs.append(clean_seq("".join(buf)))
                    chroms.append(chrom)
                m = re.search(r'(chr[^\s:()]+)', line)
                chrom, buf = (m.group(1) if m else "unknown"), []
            else:
                buf.append(line)
    if buf and chrom:
        seqs.append(clean_seq("".join(buf)))
        chroms.append(chrom)
    return seqs, chroms


def extract_all_sequences(peaks_df: pd.DataFrame, neg_bed: Path,
                          out_dir: Path) -> Tuple[List, List, List, List]:
    print("\n[3/8] Extracting sequences from GRCh38...")

    pos_fa = out_dir / "positives.fa"
    neg_fa = out_dir / "negatives.fa"

    ok = extract_via_samtools(out_dir / "peaks_cleaned.bed", pos_fa)
    if ok:
        extract_via_samtools(neg_bed, neg_fa)
        print("  Extracted via samtools (fast)")
    else:
        print("  samtools unavailable — falling back to in-memory genome")
        genome = load_genome_into_memory()
        print(f"  Genome loaded: {len(genome)} chromosomes")
        extract_from_bed_with_genome(out_dir / "peaks_cleaned.bed", pos_fa, genome)
        extract_from_bed_with_genome(neg_bed, neg_fa, genome)

    pos_seqs, pos_chroms = read_fasta_with_coords(pos_fa)
    neg_seqs, neg_chroms = read_fasta_with_coords(neg_fa)

    print(f"  Positive sequences  : {len(pos_seqs):,}")
    print(f"  Negative sequences  : {len(neg_seqs):,}")
    if pos_seqs:
        lens = [len(s) for s in pos_seqs]
        print(f"  Length range        : {min(lens)}–{max(lens)} bp  (median {int(np.median(lens))} bp)")

    return pos_seqs, pos_chroms, neg_seqs, neg_chroms


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — BUILD TRAIN / VAL / TEST SPLITS
# ─────────────────────────────────────────────────────────────────────────────

def build_splits(pos_seqs, pos_chroms, neg_seqs, neg_chroms,
                 test_chroms, val_chroms):
    """
    Split positives and negatives by chromosome into train/val/test.
    Returns dicts with keys: train, val, test
    Each value is a tuple (sequences, labels)
    """
    test_set  = set(test_chroms)
    val_set   = set(val_chroms)

    def split(seqs, chroms, label):
        tr, va, te = [], [], []
        for s, c in zip(seqs, chroms):
            if c in test_set:       te.append((s, label))
            elif c in val_set:      va.append((s, label))
            else:                   tr.append((s, label))
        return tr, va, te

    pos_tr, pos_va, pos_te = split(pos_seqs, pos_chroms, 1)
    neg_tr, neg_va, neg_te = split(neg_seqs, neg_chroms, 0)

    rng = random.Random(RANDOM_STATE)

    def balance_and_shuffle(pos, neg):
        n = min(len(pos), len(neg))
        rng.shuffle(pos); rng.shuffle(neg)
        combined = pos[:n] + neg[:n]
        rng.shuffle(combined)
        seqs   = [s for s, _ in combined]
        labels = [l for _, l in combined]
        return seqs, labels

    train_seqs, train_labels = balance_and_shuffle(pos_tr, neg_tr)
    val_seqs,   val_labels   = balance_and_shuffle(pos_va, neg_va)
    test_seqs,  test_labels  = balance_and_shuffle(pos_te, neg_te)

    print(f"  Train set           : {len(train_seqs):,} sequences "
          f"({sum(train_labels):,} pos / {len(train_labels)-sum(train_labels):,} neg)")
    print(f"  Val set             : {len(val_seqs):,} sequences")
    print(f"  Test set            : {len(test_seqs):,} sequences")

    if len(test_seqs) == 0:
        print("\n  WARNING: No sequences found on test chromosomes.")
        print(f"  Test chroms requested: {test_chroms}")
        print(f"  Chroms in data: {sorted(set(pos_chroms))}")
        print("  Consider changing --test-chroms to chroms present in your data.")
        sys.exit(1)

    return (train_seqs, train_labels), (val_seqs, val_labels), (test_seqs, test_labels)


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


def one_hot(seq: str) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((4, MAX_LEN), dtype=np.float32)
    for i, b in enumerate(seq[:MAX_LEN]):
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
              out_dir: Path, name: str) -> RBP_CNN:
    print("  Training CNN...")
    model = RBP_CNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = SeqDataset(train_seqs, train_labels)
    val_ds   = SeqDataset(val_seqs,   val_labels)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    best_val_loss = float("inf")
    best_weights  = None
    patience, patience_count = 5, 0

    for epoch in range(30):
        # Train
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
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

    # Save weights
    weights_path = out_dir / f"{name}_cnn.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"  CNN weights saved   : {weights_path}")
    return model


def cnn_predict(model: RBP_CNN, seqs: List[str]) -> List[float]:
    ds     = SeqDataset(seqs)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    model.eval()
    probs = []
    with torch.no_grad():
        for x in loader:
            logits = model(x.to(DEVICE))
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return probs


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5b — LR MODEL
# ─────────────────────────────────────────────────────────────────────────────

def seq_to_kmers(seq: str, ks=(2, 3)) -> str:
    seq = clean_seq(seq)
    tokens = []
    for k in ks:
        tokens.extend(seq[i:i+k] for i in range(len(seq) - k + 1))
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
    texts = [seq_to_kmers(s) for s in seqs]
    X     = vectorizer.transform(texts)
    return clf.predict_proba(X)[:, 1].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 (combined) — TRAIN & EVALUATE MODELS
# ─────────────────────────────────────────────────────────────────────────────

def train_and_score(train_data, val_data, test_data, out_dir, name):
    print("\n[4/8] Training and evaluating models...")

    train_seqs, train_labels = train_data
    val_seqs,   val_labels   = val_data
    test_seqs,  test_labels  = test_data

    # CNN
    cnn   = train_cnn(train_seqs, train_labels, val_seqs, val_labels, out_dir, name)
    cnn_probs = cnn_predict(cnn, test_seqs)
    cnn_auc   = roc_auc_score(test_labels, cnn_probs)
    print(f"  CNN  test AUC       : {cnn_auc:.4f}")

    # LR
    clf, vec  = train_lr(train_seqs, train_labels)
    lr_probs  = lr_predict(clf, vec, test_seqs)
    lr_auc    = roc_auc_score(test_labels, lr_probs)
    print(f"  LR   test AUC       : {lr_auc:.4f}")

    # Ensemble
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

    # ROC curve
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
    ax.set_title(f"ROC Curve — {name}", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = out_dir / "figures" / "roc_curve.png"
    plt.savefig(roc_path, dpi=150); plt.close()
    print(f"  ROC curve           : {roc_path}")

    # Score distributions
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
        ax.set_title(f"{model_name} score distribution — {name}", fontsize=11)
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


def annotate_regions(peaks_df: pd.DataFrame, out_dir: Path, skip: bool) -> pd.DataFrame:
    print("\n[6/8] Annotating genomic regions...")
    if skip or not GTF_FILE.exists():
        if skip:
            print("  Skipped (--skip-annotation)")
        else:
            print(f"  GTF not found at {GTF_FILE} — skipping")
        peaks_df["region_type"] = "not_annotated"
        peaks_df["gene_name"]   = "."
        return peaks_df

    gtf = parse_gtf(GTF_FILE)
    region_types, gene_names = [], []
    for _, row in peaks_df.iterrows():
        rt, gn = annotate_peak(row["chrom"], row["start"], row["end"], gtf)
        region_types.append(rt)
        gene_names.append(gn)

    peaks_df["region_type"] = region_types
    peaks_df["gene_name"]   = gene_names

    counts = peaks_df["region_type"].value_counts()
    print("  Region breakdown:")
    for rt, cnt in counts.items():
        print(f"    {rt:<15} : {cnt:>5,}  ({100*cnt/len(peaks_df):.1f}%)")
    return peaks_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — SAVE CSV + WORD REPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(peaks_df, results, out_dir, name) -> pd.DataFrame:
    print("\n[7/8] Saving predictions CSV...")

    # Score ALL peaks (not just test set) for the CSV output
    all_seqs = []
    for _, row in peaks_df.iterrows():
        # sequences already extracted to positives.fa — re-read inline
        all_seqs.append("")  # placeholder; filled below

    # Re-read pos sequences
    pos_fa = out_dir / "positives.fa"
    pos_seqs, _ = read_fasta_with_coords(pos_fa)

    n = min(len(peaks_df), len(pos_seqs))
    df = peaks_df.head(n).copy()
    df["sequence"] = pos_seqs[:n]
    df["seq_length"] = [len(s) for s in pos_seqs[:n]]

    # Score all peaks with trained models
    cnn_all  = cnn_predict(results["cnn_model"], pos_seqs[:n])
    lr_all   = lr_predict(results["lr_clf"], results["lr_vec"], pos_seqs[:n])
    mean_all = [(c+l)/2 for c,l in zip(cnn_all, lr_all)]

    df["cnn_prob"]        = cnn_all
    df["lr_prob"]         = lr_all
    df["mean_prob"]       = mean_all
    df["high_confidence"] = (df["cnn_prob"] > 0.5) & (df["lr_prob"] > 0.5)
    df = df.sort_values("mean_prob", ascending=False)

    csv_path = out_dir / f"{name}_predictions.csv"
    df.to_csv(csv_path, index=False)

    n_high = df["high_confidence"].sum()
    print(f"  CSV saved           : {csv_path}")
    print(f"  High-confidence     : {n_high:,} / {len(df):,} ({100*n_high/len(df):.1f}%)")
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

    cnn_auc  = results["cnn_auc"]
    lr_auc   = results["lr_auc"]
    mean_auc = results["mean_auc"]
    n_high   = int(peaks_df["high_confidence"].sum()) if "high_confidence" in peaks_df.columns else 0
    pct_high = 100 * n_high / len(peaks_df) if len(peaks_df) > 0 else 0
    interp   = interpret_auc(mean_auc)

    region_counts = peaks_df["region_type"].value_counts().to_dict() \
        if "region_type" in peaks_df.columns else {}

    top5 = peaks_df.head(5)
    top5_js = "".join(
        f"""new TableRow({{ children: [
          cell("{row['chrom']}:{row['start']}-{row['end']}", 2800),
          cell("{row.get('gene_name','.')}", 1600),
          cell("{row.get('region_type','.')}", 1400),
          cell("{row.get('cnn_prob',0):.4f}", 1000),
          cell("{row.get('lr_prob',0):.4f}", 1000),
          cell("{row.get('mean_prob',0):.4f}", 1000)
        ] }}),"""
        for _, row in top5.iterrows()
    )

    region_rows = "".join(
        f"""new TableRow({{ children: [
          cell("{rt}", 3000), cell("{cnt}", 1500),
          cell("{100*cnt/len(peaks_df):.1f}%", 1500)
        ] }}),"""
        for rt, cnt in region_counts.items()
    )

    js = f"""
const {{ Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
         AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType }} = require(require('path').join(__dirname, 'node_modules', 'docx'));
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
        new TextRun({{ text: "RBP Binding Analysis: {name}", bold: true, size: 40, font: "Arial" }})
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
          new TableRow({{ children: [cell("Protein",3600), cell("{name}",5760)] }}),
          new TableRow({{ children: [cell("Total peaks",3600), cell("{len(peaks_df):,}",5760)] }}),
          new TableRow({{ children: [cell("High-confidence peaks",3600), cell("{n_high:,} ({pct_high:.1f}%)",5760)] }}),
          new TableRow({{ children: [cell("Test chromosomes",3600), cell("{' '.join(test_chroms)}",5760)] }}),
          new TableRow({{ children: [cell("Val chromosomes",3600), cell("{' '.join(val_chroms)}",5760)] }}),
          new TableRow({{ children: [cell("CNN AUC",3600), cell("{cnn_auc:.4f}",5760)] }}),
          new TableRow({{ children: [cell("LR AUC",3600), cell("{lr_auc:.4f}",5760)] }}),
          new TableRow({{ children: [cell("Ensemble AUC",3600), cell("{mean_auc:.4f}",5760)] }}),
        ]
      }}),
      spacer(),

      h1("2. Model Performance"),
      para("{interp}"),
      spacer(),
      para("See figures/roc_curve.png for the ROC curve and figures/score_distributions.png for score distributions."),
      spacer(),

      h1("3. Top 5 Peaks by Binding Probability"),
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
      h2("5.1 Input Data"),
      para("eCLIP BED file for {name} containing {len(peaks_df):,} peaks on GRCh38."),
      spacer(),
      h2("5.2 Negative Region Generation"),
      para("Matched negative regions generated using bedtools shuffle, constrained to genic regions and excluding peak coordinates. Seed = {RANDOM_STATE}."),
      spacer(),
      h2("5.3 Train/Val/Test Split"),
      para("Chromosome-based split: training on all chromosomes except validation ({' '.join(val_chroms)}) and test ({' '.join(test_chroms)}) sets. Positive and negative sets balanced by downsampling to equal counts within each split."),
      spacer(),
      h2("5.4 CNN"),
      para("Convolutional neural network. Architecture: Conv1D(4->64, k=7) -> BN -> MaxPool(4) -> Conv1D(64->128, k=7) -> BN -> MaxPool(4) -> GlobalMaxPool -> FC(128->64, ReLU, Dropout 0.3) -> FC(64->1). Trained with Adam (lr=1e-3), BCEWithLogitsLoss, early stopping (patience=5), max 30 epochs. Input: one-hot encoded sequences, max length {MAX_LEN} bp."),
      spacer(),
      h2("5.5 Logistic Regression"),
      para("k-mer logistic regression with 2-mer and 3-mer features (CountVectorizer). Trained on the same chromosome-split training set. Max iterations = 1000."),
      spacer(),
      h2("5.6 Evaluation"),
      para("AUC (Area Under ROC Curve) computed on held-out test chromosomes using scikit-learn roc_auc_score."),
    ]
  }}]
}});

Packer.toBuffer(doc).then(buf => {{
  fs.writeFileSync('{out_dir}/{name}_report.docx', buf);
  console.log('Report saved.');
}}).catch(err => {{ console.error(err); process.exit(1); }});
"""

    js_path = out_dir / "_report.js"
    with open(js_path, "w") as f:
        f.write(js)

    # Install docx if needed
    check = subprocess.run(["node", "-e", "require(require('path').join(__dirname, 'node_modules', 'docx'))"], capture_output=True)
    if check.returncode != 0:
        print("  Installing docx npm package...")
        subprocess.run(["npm", "install", "--prefix", str(Path(__file__).parent), "docx"], check=True)

    result = subprocess.run(["node", str(js_path)], capture_output=True, text=True)
    js_path.unlink(missing_ok=True)

    if result.returncode == 0:
        print(f"  Word report saved   : {out_dir}/{name}_report.docx")
    else:
        print(f"  Warning: Word report failed: {result.stderr[:300]}")
        print("  CSV and figures were still saved.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    name = args.name
    bed_in = Path(args.input)

    print("=" * 65)
    print(f"  RBP ANALYSIS PIPELINE")
    print(f"  Protein      : {name}")
    print(f"  Input        : {bed_in}")
    print(f"  Test chroms  : {args.test_chroms}")
    print(f"  Val chroms   : {args.val_chroms}")
    print(f"  Device       : {DEVICE}")
    print("=" * 65)

    if not bed_in.exists():
        print(f"ERROR: Input file not found: {bed_in}")
        sys.exit(1)

    out_dir = setup_output_dir(name, args.output_dir)
    print(f"  Output dir   : {out_dir}")

    # Steps
    peaks_df = validate_bed(bed_in, out_dir)

    if args.negatives:
        neg_bed = Path(args.negatives)
        print(f"\n[2/8] Using provided negatives: {neg_bed}")
    else:
        neg_bed = generate_negatives(peaks_df, out_dir, name)

    pos_seqs, pos_chroms, neg_seqs, neg_chroms = extract_all_sequences(
        peaks_df, neg_bed, out_dir)

    print("\n[3/8] Building chromosome splits...")
    train_data, val_data, test_data = build_splits(
        pos_seqs, pos_chroms, neg_seqs, neg_chroms,
        args.test_chroms, args.val_chroms)

    results = train_and_score(train_data, val_data, test_data, out_dir, name)

    make_plots(results, out_dir, name)

    peaks_df = annotate_regions(peaks_df, out_dir, args.skip_annotation)

    peaks_df = save_csv(peaks_df, results, out_dir, name)

    generate_report(peaks_df, results, out_dir, name,
                    args.test_chroms, args.val_chroms)

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65)
    print(f"  CNN  AUC : {results['cnn_auc']:.4f}")
    print(f"  LR   AUC : {results['lr_auc']:.4f}")
    print(f"  Mean AUC : {results['mean_auc']:.4f}")
    print(f"\n  Results  : {out_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()