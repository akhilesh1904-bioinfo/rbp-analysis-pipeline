"""Models: a small 1D CNN on one-hot sequence, and k-mer logistic regression."""
from __future__ import annotations

import logging
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from .config import ModelConfig
from .dataset import SeqDataset, seq_to_kmers

log = logging.getLogger("rbp_pipeline")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed every random number generator the pipeline uses."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RBP_CNN(nn.Module):
    """Conv(4->64,k7)-BN-Pool(4) -> Conv(64->128,k7)-BN-Pool(4) -> global max -> FC(128->64->1).

    No reverse-complement augmentation: sequences are single-stranded RNA in a fixed orientation.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.max(x, dim=2).values
        return self.classifier(x).squeeze(1)


def train_cnn(train: Tuple[List[str], List[int]], val: Tuple[List[str], List[int]],
              cfg: ModelConfig, seed: int, device: torch.device = DEVICE) -> Tuple[RBP_CNN, List[dict]]:
    """Train with early stopping on validation loss. Returns (best model, per-epoch history)."""
    log.info("  Training CNN...")
    set_seed(seed)
    model = RBP_CNN(dropout=cfg.dropout).to(device)
    if cfg.weight_decay > 0:   # decoupled weight decay
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    loader_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(SeqDataset(*train, max_len=cfg.max_len), batch_size=cfg.batch_size,
                              shuffle=True, generator=loader_gen)
    val_loader = DataLoader(SeqDataset(*val, max_len=cfg.max_len), batch_size=cfg.batch_size, shuffle=False)

    best_val_loss, best_weights, bad_epochs = float("inf"), None, 0
    history: List[dict] = []
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()

        train_loss /= len(train_loader)
        val_loss /= max(len(val_loader), 1)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if (epoch + 1) % 5 == 0:
            log.info(f"    Epoch {epoch+1:2d} — train loss: {train_loss:.4f}  val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss, bad_epochs = val_loss, 0
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                log.info(f"    Early stopping at epoch {epoch+1}")
                break

    if best_weights:
        model.load_state_dict(best_weights)
    return model, history


def cnn_predict(model: RBP_CNN, seqs: Sequence[str], max_len: int, batch_size: int = 256,
                device: torch.device = DEVICE) -> List[float]:
    loader = DataLoader(SeqDataset(list(seqs), max_len=max_len), batch_size=batch_size, shuffle=False)
    model.eval()
    probs: List[float] = []
    with torch.no_grad():
        for x in loader:
            probs.extend(torch.sigmoid(model(x.to(device))).cpu().numpy().tolist())
    return probs


def train_lr(seqs: Sequence[str], labels: Sequence[int], seed: int):
    """k-mer (2,3) count logistic regression. The vectorizer is fit on training data only."""
    log.info("  Training LR...")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([seq_to_kmers(s) for s in seqs])
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X, np.array(labels))
    log.info(f"  LR trained on {len(seqs):,} sequences")
    return clf, vectorizer


def lr_predict(clf, vectorizer, seqs: Sequence[str]) -> List[float]:
    X = vectorizer.transform([seq_to_kmers(s) for s in seqs])
    return clf.predict_proba(X)[:, 1].tolist()
