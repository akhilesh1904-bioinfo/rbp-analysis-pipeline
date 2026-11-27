"""
Biological interpretation step: what sequence features the models use, de novo and known motifs,
and gene-set enrichment of bound genes. Tables go to <out_dir>/interpretation/, figures to
<out_dir>/figures/.

Only held-out test positives are explained by the CNN, so attributions describe sequences the
model never trained on. Motif discovery and known-motif enrichment describe the whole dataset
(positives vs their matched negatives) and are not performance estimates.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import PipelineError
from .attribution import (attribution_matrix, cnn_importance, filter_motifs, kmer_importance,
                          lr_kmer_table, position_profile)
from .config import Config
from .enrichment import read_gene_file, run_enrichment
from .logos import draw_logo, pwm_heights
from .motifs import (iupac_matches, known_motif_enrichment, meme_tool, run_streme, run_tomtom,
                     write_meme)

log = logging.getLogger("rbp_pipeline")


def run_interpretation(pos_df: pd.DataFrame, neg_df: pd.DataFrame, data, results: Dict[str, Any],
                       cfg: Config, out_dir: Path) -> Dict[str, Any]:
    log.info("\n[8/9] Interpreting the models...")
    it = cfg.interpretation
    idir = out_dir / "interpretation"
    idir.mkdir(exist_ok=True)
    fig_dir = out_dir / "figures"
    summary: Dict[str, Any] = {"method": it.method}

    cnn_kmers, motifs_for_tomtom = _cnn_section(pos_df, results, cfg, idir, fig_dir, summary)
    lr_df = _lr_section(data, results, it.top_kmers, idir, fig_dir, cfg.name, summary)
    _known_section(pos_df, neg_df, it.known_motifs, cnn_kmers, lr_df, it.top_kmers, idir, summary)
    if it.motifs:
        motifs_for_tomtom += _streme_section(pos_df, neg_df, it, idir, fig_dir, cfg.name, summary)
    else:
        summary["streme"] = {"status": "skipped: interpretation.motifs is false"}
    _tomtom_section(motifs_for_tomtom, it.motif_db, idir, summary)
    if it.enrichment:
        _enrichment_section(pos_df, neg_df, cfg, out_dir, idir, summary)
    else:
        summary["enrichment"] = {"status": "skipped: interpretation.enrichment is false"}

    (idir / "interpretation_summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    log.info(f"  Interpretation tables: {idir}")
    return summary


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ─────────────────────────────────────────────────────────────────────────────
# CNN
# ─────────────────────────────────────────────────────────────────────────────

def _cnn_section(pos_df, results, cfg, idir, fig_dir, summary):
    it = cfg.interpretation
    test_pos = pos_df[pos_df["split"] == "test"].sort_values("cnn_prob", ascending=False)
    chosen = test_pos.head(it.n_sequences)
    if chosen.empty:
        summary["cnn"] = {"status": "skipped: no held-out test positives"}
        return pd.DataFrame(), []

    model = results["cnn_model"]
    log.info(f"  CNN {it.method.upper()} on the top {len(chosen):,} held-out test positives...")
    seqs, imps = cnn_importance(model, chosen["sequence"].tolist(), cfg.model.max_len, it.method)

    width = max(len(s) for s in seqs)
    padded = np.full((len(imps), width), np.nan, dtype=np.float32)
    for n, imp in enumerate(imps):
        padded[n, :len(imp)] = imp
    np.savez_compressed(idir / "cnn_importance.npz", peak_id=chosen["peak_id"].to_numpy(dtype=str),
                        sequence=np.array(seqs), importance=padded, method=it.method)

    profile = position_profile(imps)
    profile.to_csv(idir / "cnn_position_importance.csv", index=False)
    kdf = kmer_importance(seqs, imps, it.kmer_size)
    kdf.to_csv(idir / "cnn_kmer_importance.csv", index=False)

    filters = filter_motifs(model, seqs, cfg.model.max_len)
    pd.DataFrame([{k: v for k, v in m.items() if k != "pwm"} for m in filters]) \
        .to_csv(idir / "cnn_filter_motifs.csv", index=False)
    if filters:
        write_meme([{"id": m["id"], "alt": m["consensus"], "pwm": m["pwm"], "n_sites": m["n_sites"]}
                    for m in filters], idir / "cnn_filter_motifs.meme")

    _plot_position_profile(profile, fig_dir / "interp_position_importance.png", cfg.name, it.method)
    _plot_top_sequences(chosen, seqs, imps, fig_dir / "interp_top_sequences.png", cfg.name)
    _plot_kmers(kdf.head(it.top_kmers), "mean_importance", f"Mean {it.method.upper()} importance (sum over k-mer)",
                f"CNN: most important {it.kmer_size}-mers — {cfg.name}", fig_dir / "interp_cnn_kmers.png")
    _plot_pwms(filters[:8], fig_dir / "interp_cnn_filters.png",
               lambda m: f"{m['id']}  {m['consensus']}  (influence {m['influence']:+.2f}, {m['n_sites']} sites)")

    top = kdf.head(it.top_kmers)
    summary["cnn"] = {
        "status": "ok", "n_sequences": len(chosen), "kmer_size": it.kmer_size,
        "top_kmers": top["kmer_rna"].tolist(),
        "centre_vs_flank_importance": _centre_ratio(profile),
        "top_filters": [{"id": m["id"], "consensus": m["consensus"], "influence": m["influence"],
                         "n_sites": m["n_sites"]} for m in filters[:8]],
    }
    log.info(f"  {'Top CNN ' + str(it.kmer_size) + '-mers':<20}: {', '.join(top['kmer_rna'].head(8))}")
    if filters:
        log.info(f"  Top CNN filters     : {', '.join(m['consensus'] for m in filters[:5])}")
    tomtom = [{"id": m["id"], "alt": m["consensus"], "pwm": m["pwm"], "n_sites": m["n_sites"],
               "source": "cnn_filter"} for m in filters[:16]]
    return kdf, tomtom


def _centre_ratio(profile: pd.DataFrame) -> float:
    """Mean |importance| in the central 20 bp divided by the mean outside it (>1 = signal at the peak centre)."""
    if profile.empty:
        return float("nan")
    centre = profile["offset_from_centre"].abs() <= 10
    inner = profile.loc[centre, "mean_importance"].abs().mean()
    outer = profile.loc[~centre, "mean_importance"].abs().mean()
    return float(inner / outer) if outer and outer > 0 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# LR
# ─────────────────────────────────────────────────────────────────────────────

def _lr_section(data, results, top_n, idir, fig_dir, name, summary):
    df = lr_kmer_table(results["lr_clf"], results["lr_vec"], *data["train"])
    df.to_csv(idir / "lr_kmer_coefficients.csv", index=False)
    top_pos, top_neg = df.head(top_n), df.tail(top_n).iloc[::-1]
    _plot_lr(top_pos, top_neg, fig_dir / "interp_lr_kmers.png", name)
    summary["lr"] = {"top_positive": top_pos["kmer_rna"].tolist(), "top_negative": top_neg["kmer_rna"].tolist(),
                     "kmer_sizes": sorted(set(df["k"].tolist()))}
    log.info(f"  Top LR k-mers (+)   : {', '.join(top_pos['kmer_rna'].head(8))}")
    log.info(f"  Top LR k-mers (-)   : {', '.join(top_neg['kmer_rna'].head(8))}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Known motifs, STREME, Tomtom
# ─────────────────────────────────────────────────────────────────────────────

def _known_section(pos_df, neg_df, known, cnn_kmers, lr_df, top_n, idir, summary):
    if not known:
        summary["known_motifs"] = []
        return
    df = known_motif_enrichment(pos_df["sequence"].tolist(), neg_df["sequence"].tolist(), known)
    cnn_top = cnn_kmers["kmer"].head(top_n).tolist() if len(cnn_kmers) else []
    lr_top = lr_df["kmer"].head(top_n).tolist()
    df["in_top_cnn_kmers"] = [any(iupac_matches(m, k) for k in cnn_top) for m in df["iupac"]]
    df["in_top_lr_kmers"] = [any(iupac_matches(m, k) for k in lr_top) for m in df["iupac"]]
    df.to_csv(idir / "known_motif_enrichment.csv", index=False)
    summary["known_motifs"] = df.to_dict(orient="records")
    log.info("  Known motifs (positives vs matched negatives):")
    for r in df.itertuples():
        log.info(f"    {r.motif:<16} {r.iupac:<8} {100*r.frac_pos:5.1f}% vs {100*r.frac_neg:5.1f}%  "
                 f"OR {r.odds_ratio:5.2f}  q = {r.q_value:.1e}")


def _streme_section(pos_df, neg_df, it, idir, fig_dir, name, summary) -> List[dict]:
    if not meme_tool("streme"):
        msg = "skipped: streme not on PATH (conda install -c bioconda meme)"
        log.info("  STREME skipped: streme not on PATH (conda install -c bioconda meme)")
        summary["streme"] = {"status": msg}
        return []
    log.info(f"  STREME: {len(pos_df):,} positives vs {len(neg_df):,} matched negatives...")
    try:
        motifs = run_streme(pos_df["sequence"].tolist(), neg_df["sequence"].tolist(), idir, it.streme_nmotifs)
    except PipelineError as e:
        log.warning(f"  Warning: {e}")
        summary["streme"] = {"status": f"failed: {str(e)[:200]}"}
        return []
    rows = [{"rank": i + 1, "id": m["id"], "consensus": m["consensus"], "width": len(m["pwm"]),
             "nsites": m["nsites"], "evalue": m["evalue"]} for i, m in enumerate(motifs)]
    pd.DataFrame(rows, columns=["rank", "id", "consensus", "width", "nsites", "evalue"]) \
        .to_csv(idir / "streme_motifs.csv", index=False)
    _plot_pwms(motifs[:8], fig_dir / "interp_streme_motifs.png",
               lambda m: f"{m['id']}  (E = {m['evalue']:.1e}, {int(m['nsites'] or 0)} sites)")
    summary["streme"] = {"status": "ok", "motifs": rows}
    log.info(f"  STREME motifs       : {', '.join(r['consensus'] for r in rows[:5]) or 'none'}")
    return [{"id": m["id"], "alt": m["consensus"], "pwm": m["pwm"], "n_sites": m["nsites"] or 20,
             "source": "streme"} for m in motifs]


def _tomtom_section(motifs, db, idir, summary):
    if not db:
        summary["tomtom"] = {"status": "skipped: no interpretation.motif_db"}
        return
    if not motifs:
        summary["tomtom"] = {"status": "skipped: no motifs to compare"}
        return
    if not meme_tool("tomtom"):
        summary["tomtom"] = {"status": "skipped: tomtom not on PATH"}
        log.info("  Tomtom skipped (not on PATH)")
        return
    query = write_meme(motifs, idir / "tomtom_query.meme")
    try:
        best = run_tomtom(query, Path(db), idir / "tomtom")
    except PipelineError as e:
        log.warning(f"  Warning: {e}")
        summary["tomtom"] = {"status": f"failed: {str(e)[:200]}"}
        return
    source = {m["id"]: m["source"] for m in motifs}
    cons = {m["id"]: m["alt"] for m in motifs}
    best.insert(1, "query_consensus", [cons.get(q, "") for q in best["query"]])
    best.insert(1, "query_source", [source.get(q, "") for q in best["query"]])
    best.to_csv(idir / "tomtom_best_matches.csv", index=False)
    summary["tomtom"] = {"status": "ok", "database": Path(db).name, "matches": best.to_dict(orient="records")}
    for r in best.head(8).itertuples():
        log.info(f"    {r.query:<14} {r.query_consensus:<14} -> {r.target} {r.target_name}  q = {r.q_value:.1e}")


# ─────────────────────────────────────────────────────────────────────────────
# Enrichment
# ─────────────────────────────────────────────────────────────────────────────

def _enrichment_section(pos_df, neg_df, cfg, out_dir, idir, summary):
    it = cfg.interpretation
    if "gene_name" not in pos_df or not cfg.reference.gtf or not cfg.steps.annotation:
        summary["enrichment"] = {"status": "skipped: needs reference.gtf and steps.annotation"}
        log.info("  Enrichment skipped (needs reference.gtf and steps.annotation)")
        return
    log.info("  Gene-set enrichment (g:Profiler)...")
    cache = Path(cfg.output_dir) / "_cache"
    if it.background_genes:
        universe = read_gene_file(Path(it.background_genes))
        bg_label = f"file {Path(it.background_genes).name}"
    elif it.background == "sampled":
        if "gene_name" not in neg_df:      # negatives not annotated by the region QC
            from .annotation import annotate_regions
            neg_df = annotate_regions(neg_df, Path(cfg.reference.gtf), out_dir, cache, id_col="neg_id", quiet=True)
        universe = list(pos_df["gene_name"]) + list(neg_df["gene_name"])
        bg_label = "genes overlapping peaks or negatives"
    else:
        universe, bg_label = None, "all annotated genes"
    restrict = None
    if it.protein_coding_only:
        from .annotation import protein_coding_genes
        restrict = protein_coding_genes(Path(cfg.reference.gtf), cache)
        bg_label += ", protein-coding only"
    result = run_enrichment(pos_df["gene_name"].tolist(), universe, idir, organism=it.organism,
                            sources=it.enrichment_sources, restrict_to=restrict)
    summary["enrichment"] = {**result, "background": bg_label}


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _plot_position_profile(profile, path, name, method):
    if profile.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x, y = profile["offset_from_centre"], profile["mean_importance"]
    se = profile["sd"] / np.sqrt(profile["n"].clip(lower=1))
    ax.fill_between(x, y - se, y + se, color="#2196F3", alpha=0.25, lw=0)
    ax.plot(x, y, color="#2196F3", lw=1.5)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Position relative to peak centre (nt)")
    ax.set_ylabel(f"Mean {method.upper()} importance")
    ax.set_title(f"Where the CNN looks (held-out test positives) — {name}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def _plot_top_sequences(chosen, seqs, imps, path, name, n=4):
    n = min(n, len(seqs))
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(max(10, len(seqs[0]) * 0.12), 1.6 * n), squeeze=False)
    for ax, (row, s, imp) in zip(axes[:, 0], zip(chosen.head(n).itertuples(), seqs, imps)):
        draw_logo(ax, attribution_matrix(s, imp), ylabel="importance")
        ax.set_xticks([])
        gene = getattr(row, "gene_name", ".")
        ax.set_title(f"{row.chrom}:{row.start}-{row.end}({row.strand})  {gene if gene != '.' else ''}  "
                     f"CNN p = {row.cnn_prob:.3f}", fontsize=9, loc="left")
    fig.suptitle(f"Per-base importance, top held-out test peaks — {name}", fontsize=11)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def _plot_pwms(motifs, path, title_fn):
    if not motifs:
        return
    fig, axes = plt.subplots(len(motifs), 1, figsize=(4.5, 1.5 * len(motifs)), squeeze=False)
    for ax, m in zip(axes[:, 0], motifs):
        draw_logo(ax, pwm_heights(m["pwm"]))
        ax.set_ylim(0, 2.05)
        ax.set_xticks(range(len(m["pwm"])), [str(i + 1) for i in range(len(m["pwm"]))], fontsize=7)
        ax.set_title(title_fn(m), fontsize=8, loc="left")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def _plot_kmers(df, col, xlabel, title, path):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(df) + 1.2))
    ax.barh(df["kmer_rna"][::-1], df[col][::-1], color="#2196F3")
    ax.set_xlabel(xlabel); ax.set_title(title, fontsize=10)
    ax.tick_params(axis="y", labelsize=8); ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def _plot_lr(top_pos, top_neg, path, name):
    df = pd.concat([top_pos, top_neg.iloc[::-1]])
    fig, ax = plt.subplots(figsize=(6, 0.25 * len(df) + 1.2))
    colors = ["#4CAF50" if c > 0 else "#F44336" for c in df["coefficient"][::-1]]
    ax.barh(df["kmer_rna"][::-1], df["coefficient"][::-1], color=colors)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("LR coefficient (per k-mer count)")
    ax.set_title(f"LR: k-mers most associated with binding (+) and non-binding (-) — {name}", fontsize=9)
    ax.tick_params(axis="y", labelsize=8); ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
