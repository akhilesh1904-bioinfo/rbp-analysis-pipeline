"""Train both models and score them on the held-out test chromosomes."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import torch

from .config import Config
from .dataset import SplitData
from .evaluation import auc_table, gc_scores
from .models import cnn_predict, lr_predict, train_cnn, train_lr

log = logging.getLogger("rbp_pipeline")


def train_and_score(data: SplitData, cfg: Config, out_dir: Path) -> Dict[str, Any]:
    log.info("\n[4/8] Training and evaluating models...")
    test_seqs, test_labels = data["test"]

    cnn, history = train_cnn(data["train"], data["val"], cfg.model, cfg.seed)
    weights = out_dir / f"{cfg.name}_cnn.pt"
    torch.save(cnn.state_dict(), weights)
    log.info(f"  CNN weights saved   : {weights}")
    cnn_probs = cnn_predict(cnn, test_seqs, cfg.model.max_len)

    clf, vec = train_lr(*data["train"], seed=cfg.seed)
    lr_probs = lr_predict(clf, vec, test_seqs)

    mean_probs = [(c + l) / 2 for c, l in zip(cnn_probs, lr_probs)]
    gc = gc_scores(test_seqs)
    aucs = auc_table(test_labels, {"cnn": cnn_probs, "lr": lr_probs, "mean": mean_probs, "gc": gc})

    log.info(f"  CNN  test AUC       : {aucs['cnn']:.4f}")
    log.info(f"  LR   test AUC       : {aucs['lr']:.4f}")
    log.info(f"  Ensemble test AUC   : {aucs['mean']:.4f}")
    log.info(f"  GC-only baseline AUC: {aucs['gc']:.4f}  (models should clearly beat this)")

    return {
        "cnn_auc": aucs["cnn"], "lr_auc": aucs["lr"], "mean_auc": aucs["mean"], "gc_auc": aucs["gc"],
        "cnn_probs": cnn_probs, "lr_probs": lr_probs, "mean_probs": mean_probs, "gc_scores": gc,
        "test_labels": test_labels, "test_seqs": test_seqs,
        "cnn_model": cnn, "cnn_history": history, "lr_clf": clf, "lr_vec": vec,
    }
