"""
What the models learned.

CNN
  * in-silico mutagenesis (ISM): every base of a sequence is mutated to the three alternatives and
    the change in the CNN logit is recorded. A base's importance is how much the logit drops, on
    average, when it is mutated (positive = the base supports binding).
  * saliency (gradient x input): a faster first-order approximation of the same thing.
  * importances are aggregated per position (where in the window matters) and per k-mer
    (which short sequences matter), and first-layer filters are turned into motifs.
LR
  * top positive and negative k-mer coefficients.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .dataset import fit_length, one_hot, seq_to_kmers
from .models import DEVICE, RBP_CNN

log = logging.getLogger("rbp_pipeline")

BASES = "ACGT"
_BASE_INDEX = {b: i for i, b in enumerate(BASES)}


# ─────────────────────────────────────────────────────────────────────────────
# Per-base attributions
# ─────────────────────────────────────────────────────────────────────────────

def _logits(model: RBP_CNN, x: np.ndarray, device, batch: int = 4096) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(model(torch.from_numpy(x[i:i + batch]).to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def ism(model: RBP_CNN, seqs: Sequence[str], max_len: int, device: torch.device = DEVICE,
        chunk: int = 32) -> List[np.ndarray]:
    """
    In-silico mutagenesis. Returns one (L, 4) array per sequence (L = length after centre-cropping
    to max_len): entry [i, b] = logit(base i -> b) - logit(reference). The reference base's column is 0.
    Positions with N are left at 0.
    """
    model.eval()
    out: List[np.ndarray] = []
    for start in range(0, len(seqs), chunk):
        batch_seqs = [fit_length(s, max_len) for s in seqs[start:start + chunk]]
        variants, index = [], []          # index: (seq_no, pos, base) per variant; -1 marks the reference
        for n, s in enumerate(batch_seqs):
            ref = one_hot(s, max_len)
            variants.append(ref)
            index.append((n, -1, -1))
            for i, b in enumerate(s):
                if b not in _BASE_INDEX:
                    continue
                for alt in range(4):
                    if alt == _BASE_INDEX[b]:
                        continue
                    v = ref.copy()
                    v[:, i] = 0.0
                    v[alt, i] = 1.0
                    variants.append(v)
                    index.append((n, i, alt))
        logits = _logits(model, np.stack(variants), device)
        deltas = [np.zeros((len(s), 4), dtype=np.float32) for s in batch_seqs]
        ref_logit = {n: logits[k] for k, (n, i, _) in enumerate(index) if i == -1}
        for k, (n, i, alt) in enumerate(index):
            if i >= 0:
                deltas[n][i, alt] = logits[k] - ref_logit[n]
        out.extend(deltas)
    return out


def ism_importance(delta: np.ndarray, seq: str) -> np.ndarray:
    """Per-base importance from an ISM matrix: minus the mean change over the three alternatives."""
    imp = np.zeros(len(delta), dtype=np.float32)
    for i, b in enumerate(seq[:len(delta)]):
        if b in _BASE_INDEX:
            imp[i] = -delta[i].sum() / 3.0
    return imp


def saliency(model: RBP_CNN, seqs: Sequence[str], max_len: int, device: torch.device = DEVICE,
             batch: int = 256) -> List[np.ndarray]:
    """Gradient x input of the logit: one importance value per base."""
    model.eval()
    out: List[np.ndarray] = []
    for start in range(0, len(seqs), batch):
        chunk = [fit_length(s, max_len) for s in seqs[start:start + batch]]
        x = torch.from_numpy(np.stack([one_hot(s, max_len) for s in chunk])).to(device).requires_grad_(True)
        model.zero_grad()
        model(x).sum().backward()
        attr = (x.grad * x).sum(dim=1).detach().cpu().numpy()
        out.extend(attr[n, :len(s)].astype(np.float32) for n, s in enumerate(chunk))
    return out


def cnn_importance(model: RBP_CNN, seqs: Sequence[str], max_len: int, method: str = "ism",
                   device: torch.device = DEVICE) -> Tuple[List[str], List[np.ndarray]]:
    """(sequences as scored, one per-base importance vector each)."""
    cropped = [fit_length(s, max_len) for s in seqs]
    if method == "ism":
        deltas = ism(model, cropped, max_len, device)
        return cropped, [ism_importance(d, s) for d, s in zip(deltas, cropped)]
    if method == "saliency":
        return cropped, saliency(model, cropped, max_len, device)
    raise ValueError(f"Unknown attribution method '{method}'")


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def position_profile(importances: Sequence[np.ndarray]) -> pd.DataFrame:
    """Mean and SD of importance by offset from the sequence centre."""
    acc: Dict[int, List[float]] = defaultdict(list)
    for imp in importances:
        centre = len(imp) // 2
        for i, v in enumerate(imp):
            acc[i - centre].append(float(v))
    rows = [{"offset_from_centre": k, "mean_importance": float(np.mean(v)), "sd": float(np.std(v)), "n": len(v)}
            for k, v in sorted(acc.items())]
    return pd.DataFrame(rows)


def kmer_importance(seqs: Sequence[str], importances: Sequence[np.ndarray], k: int,
                    min_count: int | None = None) -> pd.DataFrame:
    """
    Sum the per-base importance over every k-mer occurrence and average per k-mer.
    `enrichment` compares a k-mer's mean importance with the mean over all k-mer windows.
    """
    scores: Dict[str, List[float]] = defaultdict(list)
    for s, imp in zip(seqs, importances):
        if len(s) < k:
            continue
        csum = np.concatenate([[0.0], np.cumsum(imp)])
        for i in range(len(s) - k + 1):
            kmer = s[i:i + k]
            if "N" in kmer:
                continue
            scores[kmer].append(float(csum[i + k] - csum[i]))
    if not scores:
        return pd.DataFrame(columns=["kmer", "kmer_rna", "count", "mean_importance", "max_importance"])
    if min_count is None:
        min_count = max(3, len(seqs) // 100)
    rows = [{"kmer": km, "kmer_rna": km.replace("T", "U"), "count": len(v),
             "mean_importance": float(np.mean(v)), "max_importance": float(np.max(v))}
            for km, v in scores.items() if len(v) >= min_count]
    df = pd.DataFrame(rows, columns=["kmer", "kmer_rna", "count", "mean_importance", "max_importance"])
    return df.sort_values("mean_importance", ascending=False).reset_index(drop=True)


def attribution_matrix(seq: str, imp: np.ndarray) -> np.ndarray:
    """(L, 4) matrix with each position's importance on its reference base, for logo plots."""
    mat = np.zeros((len(imp), 4), dtype=np.float32)
    for i, b in enumerate(seq[:len(imp)]):
        if b in _BASE_INDEX:
            mat[i, _BASE_INDEX[b]] = imp[i]
    return mat


# ─────────────────────────────────────────────────────────────────────────────
# First-layer filters -> motifs
# ─────────────────────────────────────────────────────────────────────────────

def _first_layer(model: RBP_CNN, x: torch.Tensor) -> torch.Tensor:
    return torch.relu(model.bn1(model.conv1(x)))


def filter_motifs(model: RBP_CNN, seqs: Sequence[str], max_len: int, device: torch.device = DEVICE,
                  threshold: float = 0.5, min_sites: int = 10) -> List[dict]:
    """
    DeepBind-style filter motifs. For each first-layer filter, every window whose activation is above
    `threshold` x the filter's maximum activation is a site; sites are stacked into a position
    frequency matrix. Each filter's *influence* is the drop in mean logit over `seqs` when that
    filter's output is zeroed.
    """
    model.eval()
    width = model.conv1.kernel_size[0]
    pad = model.conv1.padding[0]
    cropped = [fit_length(s, max_len) for s in seqs]
    x = torch.from_numpy(np.stack([one_hot(s, max_len) for s in cropped])).to(device)

    with torch.no_grad():
        act = _first_layer(model, x).cpu().numpy()            # (N, F, max_len)
        base_logit = model(x).mean().item()
    n_filters = act.shape[1]
    max_act = act.max(axis=(0, 2))

    influence = np.zeros(n_filters)
    with torch.no_grad():
        for f in range(n_filters):
            def zero_filter(_m, _inp, out, f=f):
                out = out.clone()
                out[:, f, :] = 0.0
                return out
            handle = model.bn1.register_forward_hook(zero_filter)   # BN output feeds the ReLU
            influence[f] = base_logit - model(x).mean().item()
            handle.remove()

    motifs = []
    for f in range(n_filters):
        if max_act[f] <= 0:
            continue
        counts = np.zeros((width, 4))
        n_sites = 0
        for n, s in enumerate(cropped):
            hits = np.where(act[n, f, :len(s)] > threshold * max_act[f])[0]
            for p in hits:
                lo = p - pad
                if lo < 0 or lo + width > len(s):
                    continue
                window = s[lo:lo + width]
                if "N" in window:
                    continue
                for j, b in enumerate(window):
                    counts[j, _BASE_INDEX[b]] += 1
                n_sites += 1
        if n_sites < min_sites:
            continue
        pfm = (counts + 0.1) / (counts + 0.1).sum(axis=1, keepdims=True)
        motifs.append({"id": f"filter_{f + 1}", "filter": f + 1, "n_sites": n_sites,
                       "influence": float(influence[f]), "max_activation": float(max_act[f]),
                       "consensus": consensus(pfm), "information_content": float(information_content(pfm).sum()),
                       "pwm": pfm})
    motifs.sort(key=lambda m: m["influence"], reverse=True)
    return motifs


def information_content(pfm: np.ndarray) -> np.ndarray:
    """Per-position information content (bits) of a probability matrix with 4 columns."""
    p = np.clip(pfm, 1e-9, 1)
    return 2.0 + (p * np.log2(p)).sum(axis=1)


def consensus(pfm: np.ndarray, alphabet: str = "ACGU") -> str:
    """Most likely base per position; lower case where the position carries < 0.5 bits."""
    ic = information_content(pfm)
    return "".join(alphabet[int(np.argmax(r))] if ic[i] >= 0.5 else alphabet[int(np.argmax(r))].lower()
                   for i, r in enumerate(pfm))


# ─────────────────────────────────────────────────────────────────────────────
# Logistic regression
# ─────────────────────────────────────────────────────────────────────────────

def lr_kmer_table(clf, vectorizer, train_seqs: Sequence[str], train_labels: Sequence[int]) -> pd.DataFrame:
    """
    Every LR feature with its coefficient and its mean count per training positive and negative.
    Coefficients are conditional on all other k-mers (overlapping k-mers share credit), so read the
    top of the list as candidate preferences, not independent effects.
    """
    names = vectorizer.get_feature_names_out()
    coef = clf.coef_[0]
    X = vectorizer.transform([seq_to_kmers(s, getattr(vectorizer, "rbp_ks", (2, 3))) for s in train_seqs])
    y = np.asarray(train_labels)
    pos_mean = np.asarray(X[y == 1].mean(axis=0)).ravel()
    neg_mean = np.asarray(X[y == 0].mean(axis=0)).ravel()
    df = pd.DataFrame({"kmer": names, "kmer_rna": [n.replace("T", "U") for n in names],
                       "k": [len(n) for n in names], "coefficient": coef,
                       "mean_count_pos": pos_mean, "mean_count_neg": neg_mean})
    df["log2_ratio"] = np.log2((df["mean_count_pos"] + 1e-3) / (df["mean_count_neg"] + 1e-3))
    return df.sort_values("coefficient", ascending=False).reset_index(drop=True)
