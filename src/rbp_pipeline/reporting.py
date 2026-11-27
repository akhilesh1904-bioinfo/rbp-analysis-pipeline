"""Outputs: figures, predictions CSV, run metadata, Word report."""
from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from . import __version__
from .config import Config
from .evaluation import interpret_auc
from .models import cnn_predict, lr_predict

log = logging.getLogger("rbp_pipeline")

# Where the Node.js 'docx' package is installed (works for editable and regular installs)
NODE_DIR = Path(os.environ.get("RBP_PIPELINE_NODE_DIR", Path.home() / ".cache" / "rbp_pipeline" / "node"))


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(results: Dict[str, Any], out_dir: Path, name: str) -> None:
    log.info("\n[5/8] Generating plots...")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    y = np.array(results["test_labels"])

    fig, ax = plt.subplots(figsize=(7, 6))
    for key, label, color in (("cnn", "CNN", "#2196F3"), ("lr", "LR", "#4CAF50"), ("mean", "Ensemble", "#FF5722")):
        fpr, tpr, _ = roc_curve(y, results[f"{key}_probs"])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {results[f'{key}_auc']:.3f})")
    fpr, tpr, _ = roc_curve(y, results["gc_scores"])
    ax.plot(fpr, tpr, color="#777777", lw=1.5, ls=":", label=f"GC content only (AUC = {results['gc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve (held-out test chromosomes) — {name}", fontsize=12)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "roc_curve.png", dpi=150)
    plt.close()
    log.info(f"  ROC curve           : {fig_dir / 'roc_curve.png'}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, model_name in ((axes[0], "cnn", "CNN"), (axes[1], "lr", "LR")):
        probs = np.array(results[f"{key}_probs"])
        ax.hist(probs[y == 1], bins=40, alpha=0.6, color="#2196F3", label="Positive (peaks)", density=True)
        ax.hist(probs[y == 0], bins=40, alpha=0.6, color="#F44336", label="Negative (matched)", density=True)
        ax.set_xlabel(f"{model_name} binding probability", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{model_name} test-set scores — {name}", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "score_distributions.png", dpi=150)
    plt.close()
    log.info(f"  Score distributions : {fig_dir / 'score_distributions.png'}")

    history = results.get("cnn_history") or []
    if history:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="train")
        ax.plot([h["epoch"] for h in history], [h["val_loss"] for h in history], label="validation")
        ax.set_xlabel("Epoch"); ax.set_ylabel("BCE loss"); ax.set_title(f"CNN training curve — {name}")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "cnn_training_curve.png", dpi=150)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────

def save_predictions(pos_df: pd.DataFrame, results: Dict[str, Any], cfg: Config, out_dir: Path) -> pd.DataFrame:
    """
    Score every positive peak and record which split it came from. Only peaks with
    split == 'test' were unseen during training; train/val scores are optimistic.
    """
    log.info("\n[7/8] Saving predictions CSV...")
    df = pos_df.copy()
    seqs = df["sequence"].tolist()
    df["seq_length"] = [len(s) for s in seqs]
    df["cnn_prob"] = cnn_predict(results["cnn_model"], seqs, cfg.model.max_len)
    df["lr_prob"] = lr_predict(results["lr_clf"], results["lr_vec"], seqs)
    df["mean_prob"] = (df["cnn_prob"] + df["lr_prob"]) / 2
    df["seen_in_training"] = df["split"] != "test"   # val is used for early stopping
    df["high_confidence"] = (df["cnn_prob"] > 0.5) & (df["lr_prob"] > 0.5)
    df = df.sort_values("mean_prob", ascending=False).reset_index(drop=True)

    first = ["peak_id", "chrom", "start", "end", "strand", "name", "split", "seen_in_training",
             "cnn_prob", "lr_prob", "mean_prob", "high_confidence", "region_type", "gene_name"]
    cols = [c for c in first if c in df.columns] + \
           [c for c in df.columns if c not in first and c != "sequence"] + ["sequence"]
    df = df[cols]

    csv_path = out_dir / f"{cfg.name}_predictions.csv"
    df.to_csv(csv_path, index=False)
    test = df[df["split"] == "test"]
    n_high = int(test["high_confidence"].sum())
    log.info(f"  CSV saved           : {csv_path}")
    log.info(f"  High-confidence     : {n_high:,} / {len(test):,} held-out test peaks "
             f"({100*n_high/max(len(test), 1):.1f}%)")
    log.info("  Note: scores for train/val peaks (seen_in_training=True) are optimistic.")
    return df


def save_metrics(results: Dict[str, Any], out_dir: Path, name: str) -> Path:
    keys = ("cnn_auc", "lr_auc", "mean_auc", "gc_auc", "negative_strategy", "gc_positive", "gc_negative",
            "n_test")
    metrics = {k: results[k] for k in keys if k in results}
    path = out_dir / f"{name}_metrics.json"
    path.write_text(json.dumps(metrics, indent=2))
    return path


def save_run_info(cfg: Config, out_dir: Path) -> None:
    """Record the exact configuration and software versions used for this run."""
    import sklearn
    import torch
    info = {
        "rbp_pipeline_version": __version__,
        "config": cfg.to_dict(),
        "versions": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "torch": torch.__version__, "scikit-learn": sklearn.__version__,
            "bedtools": _tool_version(["bedtools", "--version"]),
            "samtools": _tool_version(["samtools", "--version"]),
        },
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))


def _tool_version(cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()[0].strip()
    except Exception:
        return "not found"


# ─────────────────────────────────────────────────────────────────────────────
# Word report
# ─────────────────────────────────────────────────────────────────────────────

def generate_word_report(peaks_df: pd.DataFrame, results: Dict[str, Any], cfg: Config, out_dir: Path) -> None:
    log.info("\n[8/8] Generating Word report...")
    js_str = json.dumps   # safely quote any text injected into the JavaScript
    name = cfg.name
    test_df = peaks_df[peaks_df["split"] == "test"]
    n_high = int(test_df["high_confidence"].sum())
    pct_high = 100 * n_high / len(test_df) if len(test_df) else 0
    region_counts = peaks_df["region_type"].value_counts().to_dict()

    top5_js = "".join(
        f"""new TableRow({{ children: [
          cell({js_str(f"{r.chrom}:{r.start}-{r.end}({r.strand})")}, 2800),
          cell({js_str(str(r.gene_name))}, 1600), cell({js_str(str(r.region_type))}, 1400),
          cell({js_str(f"{r.cnn_prob:.4f}")}, 1000), cell({js_str(f"{r.lr_prob:.4f}")}, 1000),
          cell({js_str(f"{r.mean_prob:.4f}")}, 1000)
        ] }}),"""
        for r in test_df.head(5).itertuples())
    region_rows = "".join(
        f"""new TableRow({{ children: [
          cell({js_str(str(rt))}, 3000), cell({js_str(str(cnt))}, 1500),
          cell({js_str(f"{100*cnt/len(peaks_df):.1f}%")}, 1500)
        ] }}),"""
        for rt, cnt in region_counts.items())

    def row(label, value):
        return f"new TableRow({{ children: [cell({js_str(label)},3600), cell({js_str(str(value))},5760)] }}),"

    summary_rows = "".join([
        row("Protein", name), row("Peaks with sequences", f"{len(peaks_df):,}"),
        row("Held-out test peaks", f"{len(test_df):,}"),
        row("High-confidence test peaks", f"{n_high:,} ({pct_high:.1f}%)"),
        row("Test chromosomes", " ".join(cfg.split.test_chroms)),
        row("Val chromosomes", " ".join(cfg.split.val_chroms)),
        row("CNN AUC", f"{results['cnn_auc']:.4f}"), row("LR AUC", f"{results['lr_auc']:.4f}"),
        row("Ensemble AUC", f"{results['mean_auc']:.4f}"),
        row("GC-only baseline AUC", f"{results['gc_auc']:.4f}"),
        row("Negative strategy", results.get("negative_strategy", "")),
        row("Mean GC (pos / neg)", f"{results.get('gc_positive', 0):.1f}% / {results.get('gc_negative', 0):.1f}%"),
    ])

    gc_matched = results.get("negative_strategy") == "gc-matched"
    methods = {
        "input": f"eCLIP BED file for {name}; {len(peaks_df):,} peaks with usable sequence on GRCh38."
                 + (f" Peaks resized to {cfg.peaks.window} bp around their midpoints." if cfg.peaks.window else ""),
        "neg": ("Negative regions generated with bedtools shuffle, matched to each peak's length and chromosome, "
                f"drawn from {Path(cfg.reference.genes_bed or 'the provided BED').name} and excluding peak coordinates. "
                f"Seed = {cfg.seed}. "
                + (f"For each peak, {cfg.negatives.gc_candidates} candidates were drawn and the one with the closest "
                   "GC content was kept. " if gc_matched else "GC content was not controlled. ")
                + "A GC-content-only baseline is reported alongside the models."),
        "seq": "Sequences extracted in the RNA sense: minus-strand peaks are reverse-complemented. "
               "Sequences with more than 10% ambiguous bases were removed.",
        "split": (f"Chromosome-based split: validation ({' '.join(cfg.split.val_chroms)}), test "
                  f"({' '.join(cfg.split.test_chroms)}), training on all remaining chromosomes. Positives and "
                  "negatives balanced by downsampling within each split."),
        "cnn": ("Conv1D(4->64, k=7) -> BN -> MaxPool(4) -> Conv1D(64->128, k=7) -> BN -> MaxPool(4) -> GlobalMaxPool "
                f"-> FC(128->64, ReLU, Dropout 0.3) -> FC(64->1). Adam (lr={cfg.model.learning_rate}), "
                f"BCEWithLogitsLoss, early stopping on validation loss (patience={cfg.model.patience}), max "
                f"{cfg.model.epochs} epochs. One-hot input of {cfg.model.max_len} bp (longer sequences centre-cropped)."),
        "lr": "k-mer logistic regression with 2-mer and 3-mer counts (CountVectorizer fit on training data only).",
        "eval": ("AUC computed on held-out test chromosomes only. Peak tables in this report show test-chromosome "
                 "peaks only, because scores for training peaks are optimistic."),
    }

    docx_module = NODE_DIR / "node_modules" / "docx"
    report_path = out_dir / f"{name}_report.docx"
    js = _REPORT_JS.format(
        docx=js_str(str(docx_module)), title=js_str("RBP Binding Analysis: " + name),
        summary_rows=summary_rows, interp=js_str(interpret_auc(results["mean_auc"])),
        top5=top5_js, region_rows=region_rows, out=js_str(str(report_path)),
        **{f"m_{k}": js_str(v) for k, v in methods.items()})

    try:
        if subprocess.run(["node", "-e", f"require({js_str(str(docx_module))})"], capture_output=True).returncode:
            log.info("  Installing docx npm package...")
            NODE_DIR.mkdir(parents=True, exist_ok=True)
            subprocess.run(["npm", "install", "--prefix", str(NODE_DIR), "docx"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        log.warning(f"  Warning: could not set up Node.js 'docx' package ({e}); skipping Word report.")
        return

    js_path = out_dir / "_report.js"
    js_path.write_text(js)
    result = subprocess.run(["node", str(js_path)], capture_output=True, text=True)
    js_path.unlink(missing_ok=True)
    if result.returncode == 0:
        log.info(f"  Word report saved   : {report_path}")
    else:
        log.warning(f"  Warning: Word report failed: {result.stderr[:300]}")


_REPORT_JS = """
const {{ Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
         AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType }} = require({docx});
const fs = require('fs');
const border  = {{ style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" }};
const borders = {{ top: border, bottom: border, left: border, right: border }};
function cell(text, w, bold=false, bg="FFFFFF") {{
  return new TableCell({{ borders, width: {{ size: w, type: WidthType.DXA }},
    shading: {{ fill: bg, type: ShadingType.CLEAR }}, margins: {{ top: 80, bottom: 80, left: 120, right: 120 }},
    children: [new Paragraph({{ children: [new TextRun({{ text: String(text), bold, size: 20 }})] }})] }});
}}
function hcell(text, w) {{ return cell(text, w, true, "D5E8F0"); }}
function h1(text) {{ return new Paragraph({{ heading: HeadingLevel.HEADING_1, children: [new TextRun({{ text, bold: true, size: 32, font: "Arial" }})] }}); }}
function h2(text) {{ return new Paragraph({{ heading: HeadingLevel.HEADING_2, children: [new TextRun({{ text, bold: true, size: 26, font: "Arial" }})] }}); }}
function para(text) {{ return new Paragraph({{ children: [new TextRun({{ text, size: 22, font: "Arial" }})] }}); }}
function spacer() {{ return new Paragraph({{ children: [new TextRun("")] }}); }}
const doc = new Document({{
  styles: {{ default: {{ document: {{ run: {{ font: "Arial", size: 22 }} }} }},
    paragraphStyles: [
      {{ id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
         run: {{ size: 32, bold: true, font: "Arial" }}, paragraph: {{ spacing: {{ before: 240, after: 120 }}, outlineLevel: 0 }} }},
      {{ id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
         run: {{ size: 26, bold: true, font: "Arial" }}, paragraph: {{ spacing: {{ before: 180, after: 80 }}, outlineLevel: 1 }} }} ] }},
  sections: [{{
    properties: {{ page: {{ size: {{ width: 12240, height: 15840 }}, margin: {{ top: 1440, right: 1440, bottom: 1440, left: 1440 }} }} }},
    children: [
      new Paragraph({{ alignment: AlignmentType.CENTER, children: [new TextRun({{ text: {title}, bold: true, size: 40, font: "Arial" }})] }}),
      new Paragraph({{ alignment: AlignmentType.CENTER, children: [new TextRun({{ text: "Sequence-Based Binding Site Analysis Pipeline", size: 26, color: "555555", font: "Arial" }})] }}),
      spacer(),
      h1("1. Summary"),
      new Table({{ width: {{ size: 9360, type: WidthType.DXA }}, columnWidths: [3600, 5760],
        rows: [ new TableRow({{ children: [hcell("Parameter",3600), hcell("Value",5760)] }}), {summary_rows} ] }}),
      spacer(),
      h1("2. Model Performance"), para({interp}), spacer(),
      para("See figures/roc_curve.png (with GC-only baseline), figures/score_distributions.png and figures/cnn_training_curve.png."),
      spacer(),
      h1("3. Top 5 Held-Out Test Peaks by Binding Probability"),
      new Table({{ width: {{ size: 9360, type: WidthType.DXA }}, columnWidths: [2800, 1600, 1400, 1000, 1000, 1000],
        rows: [ new TableRow({{ children: [hcell("Coordinates",2800), hcell("Gene",1600), hcell("Region",1400),
            hcell("CNN",1000), hcell("LR",1000), hcell("Mean",1000)] }}), {top5} ] }}),
      spacer(),
      h1("4. Genomic Region Annotation"),
      new Table({{ width: {{ size: 6000, type: WidthType.DXA }}, columnWidths: [3000, 1500, 1500],
        rows: [ new TableRow({{ children: [hcell("Region",3000), hcell("Count",1500), hcell("%",1500)] }}), {region_rows} ] }}),
      spacer(),
      h1("5. Methods"),
      h2("5.1 Input Data"), para({m_input}), spacer(),
      h2("5.2 Negative Region Generation"), para({m_neg}), spacer(),
      h2("5.3 Sequence Extraction"), para({m_seq}), spacer(),
      h2("5.4 Train/Val/Test Split"), para({m_split}), spacer(),
      h2("5.5 CNN"), para({m_cnn}), spacer(),
      h2("5.6 Logistic Regression"), para({m_lr}), spacer(),
      h2("5.7 Evaluation"), para({m_eval}),
    ] }}]
}});
Packer.toBuffer(doc).then(buf => {{ fs.writeFileSync({out}, buf); console.log('Report saved.'); }})
  .catch(err => {{ console.error(err); process.exit(1); }});
"""
