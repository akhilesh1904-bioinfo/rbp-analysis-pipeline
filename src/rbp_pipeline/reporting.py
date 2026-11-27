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
    log.info("\n[5/9] Generating plots...")
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

    _extra_plots(results, fig_dir, name)

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


def _extra_plots(results: Dict[str, Any], fig_dir: Path, name: str) -> None:
    """Precision-recall curves, calibration curves and confusion matrices."""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve

    y = np.array(results["test_labels"])
    metrics = results.get("metrics", {})
    models = (("cnn", "CNN", "#2196F3"), ("lr", "LR", "#4CAF50"), ("mean", "Ensemble", "#FF5722"))

    fig, ax = plt.subplots(figsize=(7, 6))
    for key, label, color in models:
        prec, rec, _ = precision_recall_curve(y, results[f"{key}_probs"])
        ap = metrics.get("ensemble" if key == "mean" else key, {}).get("pr_auc", float("nan"))
        ax.plot(rec, prec, color=color, lw=2, label=f"{label} (AP = {ap:.3f})")
    prec, rec, _ = precision_recall_curve(y, results["gc_scores"])
    ax.plot(rec, prec, color="#777777", lw=1.5, ls=":",
            label=f"GC content only (AP = {metrics.get('gc_baseline', {}).get('pr_auc', float('nan')):.3f})")
    ax.axhline(y.mean(), color="k", ls="--", lw=1, alpha=0.5, label=f"Chance ({y.mean():.2f})")
    ax.set_xlabel("Recall", fontsize=12); ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-recall (held-out test) — {name}", fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.legend(loc="lower left", fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(fig_dir / "pr_curve.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    for key, label, color in models:
        frac_pos, mean_pred = calibration_curve(y, results[f"{key}_probs"], n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", color=color, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability", fontsize=12); ax.set_ylabel("Fraction of binding sites", fontsize=12)
    ax.set_title(f"Calibration (held-out test) — {name}", fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(fig_dir / "calibration.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (key, label, _) in zip(axes, models):
        c = metrics.get("ensemble" if key == "mean" else key, {}).get("confusion")
        if not c:
            ax.axis("off"); continue
        mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(mat, cmap="Blues")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > mat.max() / 2 else "black")
        ax.set_xticks([0, 1], ["pred non-binding", "pred binding"]); ax.set_yticks([0, 1], ["non-binding", "binding"])
        thr = metrics.get("cnn", {}).get("threshold", 0.5)
        ax.set_title(f"{label} (threshold {thr})")
    plt.tight_layout(); plt.savefig(fig_dir / "confusion_matrices.png", dpi=150); plt.close()
    log.info(f"  PR, calibration and confusion-matrix figures saved to {fig_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────

def save_predictions(pos_df: pd.DataFrame, results: Dict[str, Any], cfg: Config, out_dir: Path) -> pd.DataFrame:
    """
    Score every positive peak and record which split it came from. Only peaks with
    split == 'test' were unseen during training; train/val scores are optimistic.
    """
    log.info("\n[7/9] Saving predictions CSV...")
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
    """<name>_metrics.json (everything) and <name>_metrics.csv (one row per model x metric)."""
    keys = ("cnn_auc", "lr_auc", "mean_auc", "gc_auc", "negative_strategy", "gc_positive", "gc_negative",
            "n_test", "seeds", "metrics", "comparisons", "seed_summary", "leakage", "interpretation")
    payload = {k: results[k] for k in keys if k in results}
    path = out_dir / f"{name}_metrics.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default))

    rows = []
    seeds = results.get("seed_summary") or {}
    for model, m in results.get("metrics", {}).items():
        for metric, value in m.items():
            if not isinstance(value, (int, float)) or metric in ("n", "n_pos", "threshold"):
                continue
            ci = m.get(f"{metric}_ci") or [None, None]
            sd = seeds.get(model, {}).get(metric, {})
            rows.append({"model": model, "metric": metric, "value": value, "ci_low": ci[0], "ci_high": ci[1],
                         "seed_mean": sd.get("mean"), "seed_sd": sd.get("sd")})
    pd.DataFrame(rows).to_csv(out_dir / f"{name}_metrics.csv", index=False)
    return path


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (np.ndarray, Path)):
        return o.tolist() if isinstance(o, np.ndarray) else str(o)
    raise TypeError(f"Not JSON serialisable: {type(o).__name__}")


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

def _metric_rows(results: Dict[str, Any], row) -> list:
    m = results.get("metrics", {})
    out = []
    for key, label in (("cnn", "CNN"), ("lr", "LR")):
        if key in m:
            ci = m[key].get("roc_auc_ci") or [float("nan")] * 2
            out.append(row(f"{label} ROC-AUC (95% CI)", f"{m[key]['roc_auc']:.3f} [{ci[0]:.3f}-{ci[1]:.3f}]"))
            out.append(row(f"{label} PR-AUC / F1 / MCC",
                           f"{m[key]['pr_auc']:.3f} / {m[key]['f1']:.3f} / {m[key]['mcc']:.3f}"))
    d = (results.get("comparisons") or {}).get("cnn_vs_lr")
    if d:
        out.append(row("CNN - LR ROC-AUC (95% CI)", f"{d['diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"))
    ss = results.get("seed_summary")
    if ss:
        out.append(row(f"CNN ROC-AUC over {len(results['seeds'])} seeds",
                       f"{ss['cnn']['roc_auc']['mean']:.3f} ± {ss['cnn']['roc_auc']['sd']:.3f}"))
    lk = results.get("leakage")
    if lk:
        t = lk["test"]
        out.append(row("Test sequences resembling training",
                       f"{t['n_flagged']:,} / {t['n_sequences']:,} ({100*t['fraction_flagged']:.1f}%; mode: {lk['mode']})"))
    return out


def generate_word_report(peaks_df: pd.DataFrame, results: Dict[str, Any], cfg: Config, out_dir: Path) -> None:
    log.info("\n[9/9] Generating Word report...")
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
        row("Ensemble AUC", f"{results['mean_auc']:.4f}"),
        row("GC-only baseline AUC", f"{results['gc_auc']:.4f}"),
        *_metric_rows(results, row),
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
        "lr": (f"k-mer logistic regression on {'/'.join(map(str, cfg.model.lr_kmers))}-mer counts "
               "(CountVectorizer fit on training data only)."),
        "interp": _interp_methods(cfg) if results.get("interpretation") else "Interpretation step not run.",
        "eval": ("Metrics computed on held-out test chromosomes only: ROC-AUC, PR-AUC (average precision), and "
                 f"precision, recall, F1, MCC and Brier score at threshold {cfg.evaluation.threshold}, with "
                 f"{cfg.evaluation.bootstrap} bootstrap resamples for 95% CIs"
                 + (f" and {cfg.evaluation.n_seeds} training seeds" if cfg.evaluation.n_seeds > 1 else "")
                 + f". Validation/test sequences resembling training sequences (identical, reverse complement or a "
                 f"shared {cfg.split.leakage_kmer}-mer) were {'removed' if cfg.split.leakage == 'remove' else 'counted'}. "
                 "AUC computed on held-out test chromosomes only. Peak tables in this report show test-chromosome "
                 "peaks only, because scores for training peaks are optimistic."),
    }

    docx_module = NODE_DIR / "node_modules" / "docx"
    report_path = out_dir / f"{name}_report.docx"
    js = _REPORT_JS.format(
        docx=js_str(str(docx_module)), title=js_str("RBP Binding Analysis: " + name),
        summary_rows=summary_rows, interp=js_str(interpret_auc(results["mean_auc"])),
        top5=top5_js, region_rows=region_rows, out=js_str(str(report_path)),
        interp_paras=_interp_paras(results.get("interpretation"), js_str),
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


def _interp_paras(summary, js_str) -> str:
    """JavaScript para(...) calls summarising the interpretation step for the Word report."""
    if not summary:
        return 'para("Interpretation step not run (steps.interpretation is false)."),'
    lines = []
    cnn = summary.get("cnn", {})
    if cnn.get("status") == "ok":
        lines.append(f"CNN ({summary['method'].upper()}, {cnn['n_sequences']} top held-out test peaks): most important "
                     f"{cnn['kmer_size']}-mers {', '.join(cnn['top_kmers'][:8])}. Importance at the peak centre vs "
                     f"flanks: {cnn['centre_vs_flank_importance']:.1f}x.")
        if cnn.get("top_filters"):
            lines.append("Most influential first-layer CNN filters: "
                         + ", ".join(f"{f['consensus']} ({f['influence']:+.2f})" for f in cnn["top_filters"][:5]) + ".")
    lr = summary.get("lr")
    if lr:
        lines.append(f"Logistic regression k-mers favouring binding: {', '.join(lr['top_positive'][:8])}; "
                     f"disfavouring: {', '.join(lr['top_negative'][:8])}.")
    known = summary.get("known_motifs") or []
    if known:
        lines.append("Known motifs, % positives vs % matched negatives (odds ratio, q): " + "; ".join(
            f"{k['motif']} {100*k['frac_pos']:.1f}% vs {100*k['frac_neg']:.1f}% ({k['odds_ratio']:.2f}, {k['q_value']:.1e})"
            for k in known) + ".")
    st = summary.get("streme", {})
    lines.append("STREME de novo motifs: " + (", ".join(f"{m['consensus']} (E = {m['evalue']:.1e})"
                 for m in st["motifs"][:6]) if st.get("status") == "ok" and st.get("motifs")
                 else st.get("status", "not run")) + ".")
    tt = summary.get("tomtom", {})
    if tt.get("status") == "ok":
        lines.append(f"Tomtom best matches in {tt['database']}: " + (", ".join(
            f"{m['query_consensus']} -> {m['target_name'] or m['target']} (q = {m['q_value']:.1e})"
            for m in tt["matches"][:6]) or "none below the threshold") + ".")
    en = summary.get("enrichment", {})
    if en.get("status") == "ok":
        lines.append(f"g:Profiler ({en['n_query']} bound genes vs {en['background']}): " + (", ".join(
            f"{t['name']} ({t['source']}, p = {t['p_value']:.1e})" for t in en["top_terms"][:5])
            or "no significant terms") + ".")
    lines.append("Figures: interp_*.png. Tables: interpretation/ folder.")
    return "".join(f"para({js_str(l)}), spacer()," for l in lines)


def _interp_methods(cfg: Config) -> str:
    it = cfg.interpretation
    how = ("in-silico mutagenesis (each base mutated to the three alternatives; importance = mean drop in the "
           "CNN logit)" if it.method == "ism" else "gradient x input saliency of the CNN logit")
    return (f"CNN attributions by {how} on the {it.n_sequences} highest-scoring held-out test positives, aggregated "
            f"by position relative to the peak centre and over {it.kmer_size}-mers. First-layer filters were "
            "converted to position frequency matrices from windows activating above half the filter's maximum; "
            "filter influence is the drop in mean logit when the filter is zeroed. LR coefficients were ranked "
            "directly (they are conditional on overlapping k-mers). Known IUPAC motifs were tested for presence in "
            "positives vs matched negatives (Fisher exact test, Benjamini-Hochberg). De novo motifs: STREME (MEME "
            "Suite, RNA alphabet) on positives vs the matched negatives, compared with known motifs by Tomtom "
            "(Pearson, q < 0.1). Gene-set enrichment: g:Profiler g:GOSt (g:SCS) of genes with peaks against "
            + ("a user-supplied background gene list." if it.background_genes else
               "genes overlapped by peaks or negatives." if it.background == "sampled" else "all annotated genes."))


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
      para("Figures: roc_curve.png and pr_curve.png (with GC-only baseline), calibration.png, confusion_matrices.png, score_distributions.png, cnn_training_curve.png. Full metrics with confidence intervals: <name>_metrics.csv."),
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
      h1("5. Biological Interpretation"), {interp_paras}
      spacer(),
      h1("6. Methods"),
      h2("6.1 Input Data"), para({m_input}), spacer(),
      h2("6.2 Negative Region Generation"), para({m_neg}), spacer(),
      h2("6.3 Sequence Extraction"), para({m_seq}), spacer(),
      h2("6.4 Train/Val/Test Split"), para({m_split}), spacer(),
      h2("6.5 CNN"), para({m_cnn}), spacer(),
      h2("6.6 Logistic Regression"), para({m_lr}), spacer(),
      h2("6.7 Evaluation"), para({m_eval}), spacer(),
      h2("6.8 Interpretation"), para({m_interp}),
    ] }}]
}});
Packer.toBuffer(doc).then(buf => {{ fs.writeFileSync({out}, buf); console.log('Report saved.'); }})
  .catch(err => {{ console.error(err); process.exit(1); }});
"""
