"""Train both models (optionally over several seeds) and evaluate them on the held-out test set."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from .config import Config
from .dataset import SplitData
from .evaluation import (bootstrap_ci, compute_metrics, gc_scores, paired_bootstrap_diff,
                         summarise_seeds)
from .models import cnn_predict, lr_predict, train_cnn, train_lr

log = logging.getLogger("rbp_pipeline")

MODELS = ("cnn", "lr", "ensemble")


def _train_once(data: SplitData, cfg: Config, seed: int) -> Dict[str, Any]:
    test_seqs, _ = data["test"]
    cnn, history = train_cnn(data["train"], data["val"], cfg.model, seed)
    clf, vec = train_lr(*data["train"], seed=seed, ks=cfg.model.lr_kmers)
    cnn_probs = cnn_predict(cnn, test_seqs, cfg.model.max_len)
    lr_probs = lr_predict(clf, vec, test_seqs)
    return {"cnn_model": cnn, "cnn_history": history, "lr_clf": clf, "lr_vec": vec,
            "cnn_probs": cnn_probs, "lr_probs": lr_probs,
            "mean_probs": [(c + l) / 2 for c, l in zip(cnn_probs, lr_probs)]}


def train_and_score(data: SplitData, cfg: Config, out_dir: Path,
                    test_groups: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """test_groups: one bootstrap unit label per test sequence (None = resample sequences)."""
    log.info("\n[4/9] Training and evaluating models...")
    test_seqs, labels = data["test"]
    ev = cfg.evaluation
    seeds = [cfg.seed + i for i in range(ev.n_seeds)]

    per_seed: List[Dict[str, Dict]] = []
    primary: Dict[str, Any] = {}
    for i, seed in enumerate(seeds):
        if len(seeds) > 1:
            log.info(f"  ── seed {seed} ({i + 1}/{len(seeds)})")
        run = _train_once(data, cfg, seed)
        per_seed.append({m: compute_metrics(labels, run[f"{'mean' if m == 'ensemble' else m}_probs"], ev.threshold)
                         for m in MODELS})
        if i == 0:
            primary = run

    weights = out_dir / f"{cfg.name}_cnn.pt"
    torch.save(primary["cnn_model"].state_dict(), weights)
    log.info(f"  CNN weights saved   : {weights} (seed {seeds[0]})")

    gc = gc_scores(test_seqs)
    metrics = dict(per_seed[0])
    metrics["gc_baseline"] = compute_metrics(labels, gc, threshold=None)

    scores = {"cnn": primary["cnn_probs"], "lr": primary["lr_probs"], "ensemble": primary["mean_probs"],
              "gc_baseline": gc}
    for name, s in scores.items():
        for metric in ("roc_auc", "pr_auc"):
            lo, hi = bootstrap_ci(labels, s, metric, ev.bootstrap, seed=cfg.seed, groups=test_groups)
            metrics[name][f"{metric}_ci"] = [lo, hi]

    comparisons = {
        "cnn_vs_lr": paired_bootstrap_diff(labels, scores["cnn"], scores["lr"], ev.bootstrap, cfg.seed, test_groups),
        "cnn_vs_gc_baseline": paired_bootstrap_diff(labels, scores["cnn"], gc, ev.bootstrap, cfg.seed, test_groups),
        "lr_vs_gc_baseline": paired_bootstrap_diff(labels, scores["lr"], gc, ev.bootstrap, cfg.seed, test_groups),
    }
    seed_summary = summarise_seeds(per_seed) if len(seeds) > 1 else None

    n_units = len(set(test_groups)) if test_groups is not None else len(labels)
    if ev.bootstrap > 0:
        log.info(f"  Bootstrap unit      : {ev.bootstrap_unit if test_groups is not None else 'sequence'} "
                 f"({n_units:,} units resampled from {len(labels):,} test sequences)")
    _log_metrics(metrics, comparisons, seed_summary, ev.bootstrap > 0)

    return {
        # headline numbers (first seed) — kept for scripts that read these keys
        "cnn_auc": metrics["cnn"]["roc_auc"], "lr_auc": metrics["lr"]["roc_auc"],
        "mean_auc": metrics["ensemble"]["roc_auc"], "gc_auc": metrics["gc_baseline"]["roc_auc"],
        "metrics": metrics, "comparisons": comparisons,
        "bootstrap": {"unit": ev.bootstrap_unit if test_groups is not None else "sequence",
                      "n_units": n_units, "n_resamples": ev.bootstrap}, "seed_summary": seed_summary, "seeds": seeds,
        "cnn_probs": primary["cnn_probs"], "lr_probs": primary["lr_probs"], "mean_probs": primary["mean_probs"],
        "gc_scores": gc, "test_labels": labels, "test_seqs": test_seqs,
        "cnn_model": primary["cnn_model"], "cnn_history": primary["cnn_history"],
        "lr_clf": primary["lr_clf"], "lr_vec": primary["lr_vec"],
    }


def _log_metrics(metrics: Dict, comparisons: Dict, seed_summary, with_ci: bool) -> None:
    log.info(f"  CNN  test AUC       : {metrics['cnn']['roc_auc']:.4f}")
    log.info(f"  LR   test AUC       : {metrics['lr']['roc_auc']:.4f}")
    log.info(f"  Ensemble test AUC   : {metrics['ensemble']['roc_auc']:.4f}")
    log.info(f"  GC-only baseline AUC: {metrics['gc_baseline']['roc_auc']:.4f}  (models should clearly beat this)")

    log.info("\n  Test-set metrics" + (" (95% bootstrap CI)" if with_ci else "") + ":")
    log.info(f"    {'model':<12}{'ROC-AUC':<24}{'PR-AUC':<24}{'F1':>6}{'MCC':>7}{'Brier':>7}")
    for name in ("cnn", "lr", "ensemble", "gc_baseline"):
        m = metrics[name]

        def fmt(key):
            ci = m.get(f"{key}_ci")
            return f"{m[key]:.3f}" + (f" [{ci[0]:.3f}-{ci[1]:.3f}]" if with_ci and ci else "")
        extra = (f"{m['f1']:>6.3f}{m['mcc']:>7.3f}{m['brier']:>7.3f}" if "f1" in m else f"{'—':>6}{'—':>7}{'—':>7}")
        log.info(f"    {name:<12}{fmt('roc_auc'):<24}{fmt('pr_auc'):<24}{extra}")

    if with_ci:
        log.info("\n  ROC-AUC differences (paired bootstrap 95% CI):")
        for key, d in comparisons.items():
            verdict = "significant" if d["ci_low"] > 0 or d["ci_high"] < 0 else "not significant"
            log.info(f"    {key:<20}: {d['diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]  {verdict}")

    if seed_summary:
        n = len(next(iter(seed_summary["cnn"].values()))["values"])
        log.info(f"\n  Across {n} seeds (mean ± SD):")
        for name in MODELS:
            s = seed_summary[name]
            log.info(f"    {name:<12}ROC-AUC {s['roc_auc']['mean']:.3f} ± {s['roc_auc']['sd']:.3f}   "
                     f"PR-AUC {s['pr_auc']['mean']:.3f} ± {s['pr_auc']['sd']:.3f}   "
                     f"MCC {s['mcc']['mean']:.3f} ± {s['mcc']['sd']:.3f}")
        log.info("    (LR training is deterministic, so its SD is 0; seed variation reflects the CNN.)")
