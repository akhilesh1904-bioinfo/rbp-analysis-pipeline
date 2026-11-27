"""
Motifs: known-motif enrichment (no extra software), MEME-format I/O, and wrappers around
MEME Suite STREME (de novo discovery, positives vs matched negatives) and Tomtom (comparison
with a database of known RBP motifs).
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from . import PipelineError

log = logging.getLogger("rbp_pipeline")

_IUPAC = {"A": "A", "C": "C", "G": "G", "T": "T", "U": "T", "R": "[AG]", "Y": "[CT]", "S": "[CG]",
          "W": "[AT]", "K": "[GT]", "M": "[AC]", "B": "[CGT]", "D": "[AGT]", "H": "[ACT]", "V": "[ACG]",
          "N": "[ACGT]"}


# ─────────────────────────────────────────────────────────────────────────────
# Known motifs
# ─────────────────────────────────────────────────────────────────────────────

def iupac_regex(motif: str) -> re.Pattern:
    """IUPAC (RNA or DNA letters) -> regex over DNA-letter sequences, with overlapping matches."""
    body = "".join(_IUPAC[c] for c in motif.upper())
    return re.compile(f"(?=({body}))")


def iupac_matches(motif: str, kmer: str) -> bool:
    """True if the IUPAC motif occurs anywhere in `kmer` (both in DNA or RNA letters)."""
    return bool(iupac_regex(motif).search(kmer.upper().replace("U", "T")))


def bh_adjust(p: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def known_motif_enrichment(pos_seqs: Sequence[str], neg_seqs: Sequence[str],
                           motifs: Dict[str, str]) -> pd.DataFrame:
    """
    For each IUPAC motif: fraction of positives and of negatives containing it, sites per kb,
    odds ratio and two-sided Fisher exact p (BH-adjusted across motifs).
    """
    rows = []
    pos_bp = max(sum(len(s) for s in pos_seqs), 1)
    neg_bp = max(sum(len(s) for s in neg_seqs), 1)
    for name, motif in motifs.items():
        rx = iupac_regex(motif)
        pos_hits = [len(rx.findall(s)) for s in pos_seqs]
        neg_hits = [len(rx.findall(s)) for s in neg_seqs]
        a, b = sum(h > 0 for h in pos_hits), sum(h > 0 for h in neg_hits)
        c, d = len(pos_seqs) - a, len(neg_seqs) - b
        odds, p = fisher_exact([[a, c], [b, d]])
        rows.append({"motif": name, "iupac": motif.upper().replace("T", "U"),
                     "frac_pos": a / max(len(pos_seqs), 1), "frac_neg": b / max(len(neg_seqs), 1),
                     "sites_per_kb_pos": 1000 * sum(pos_hits) / pos_bp,
                     "sites_per_kb_neg": 1000 * sum(neg_hits) / neg_bp,
                     "odds_ratio": float(odds), "p_value": float(p)})
    df = pd.DataFrame(rows, columns=["motif", "iupac", "frac_pos", "frac_neg", "sites_per_kb_pos",
                                     "sites_per_kb_neg", "odds_ratio", "p_value"])
    df["q_value"] = bh_adjust(df["p_value"].tolist())
    return df.sort_values("odds_ratio", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# MEME minimal motif format
# ─────────────────────────────────────────────────────────────────────────────

def write_meme(motifs: Sequence[dict], path: Path, alphabet: str = "ACGU") -> Path:
    """motifs: dicts with 'id', 'pwm' (w x 4) and optional 'alt', 'n_sites'."""
    lines = ["MEME version 4", "", f"ALPHABET= {alphabet}", "", "strands: +", "",
             "Background letter frequencies", " ".join(f"{b} 0.25" for b in alphabet), ""]
    for m in motifs:
        pwm = np.asarray(m["pwm"])
        lines.append(f"MOTIF {m['id']}" + (f" {m['alt']}" if m.get("alt") else ""))
        lines.append(f"letter-probability matrix: alength= 4 w= {len(pwm)} nsites= {int(m.get('n_sites', 20))} E= 0")
        lines.extend(" ".join(f"{v:.6f}" for v in row) for row in pwm)
        lines.append("")
    Path(path).write_text("\n".join(lines))
    return Path(path)


_MATRIX = re.compile(r"letter-probability matrix:(.*)")


def read_meme(path: Path) -> List[dict]:
    """Parse a MEME-format file (STREME output, Tomtom databases). Returns id, alt, pwm, nsites, evalue."""
    motifs: List[dict] = []
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("MOTIF"):
            parts = line.split()
            current = {"id": parts[1], "alt": parts[2] if len(parts) > 2 else "", "pwm": None,
                       "nsites": None, "evalue": None}
            motifs.append(current)
        else:
            m = _MATRIX.match(line)
            if m and motifs:
                fields = dict(re.findall(r"(\w+)=\s*(\S+)", m.group(1)))
                w = int(fields.get("w", 0))
                rows = []
                j = i + 1
                while len(rows) < w and j < len(lines):
                    vals = lines[j].split()
                    if vals:
                        rows.append([float(v) for v in vals])
                    j += 1
                motifs[-1]["pwm"] = np.array(rows)
                motifs[-1]["nsites"] = float(fields["nsites"]) if "nsites" in fields else None
                motifs[-1]["evalue"] = float(fields["E"]) if "E" in fields else None
                i = j - 1
        i += 1
    return [m for m in motifs if m["pwm"] is not None and len(m["pwm"])]


# ─────────────────────────────────────────────────────────────────────────────
# MEME Suite
# ─────────────────────────────────────────────────────────────────────────────

def meme_tool(name: str) -> Optional[str]:
    return shutil.which(name)


def write_rna_fasta(seqs: Sequence[str], path: Path, prefix: str) -> Path:
    with open(path, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">{prefix}{i}\n{s.replace('T', 'U')}\n")
    return Path(path)


def run_streme(pos_seqs: Sequence[str], neg_seqs: Sequence[str], out_dir: Path, nmotifs: int,
               min_width: int = 4, max_width: int = 12) -> List[dict]:
    """
    De novo motifs enriched in positives relative to the matched negatives (single-stranded RNA
    alphabet). Returns the parsed motifs (id, consensus, pwm, nsites, evalue) or raises PipelineError.
    """
    exe = meme_tool("streme")
    if not exe:
        raise PipelineError("streme not found on PATH (install MEME Suite: conda install -c bioconda meme)")
    out_dir.mkdir(parents=True, exist_ok=True)
    pos_fa = write_rna_fasta(pos_seqs, out_dir / "streme_positives.fa", "pos")
    neg_fa = write_rna_fasta(neg_seqs, out_dir / "streme_negatives.fa", "neg")
    cmd = [exe, "--p", str(pos_fa), "--n", str(neg_fa), "--rna", "--nmotifs", str(nmotifs),
           "--minw", str(min_width), "--maxw", str(max_width), "--oc", str(out_dir / "streme"),
           "--verbosity", "1"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise PipelineError(f"STREME failed:\n{e.stderr[-1000:]}") from e
    result = out_dir / "streme" / "streme.txt"
    if not result.exists():
        raise PipelineError(f"STREME finished but {result} is missing")
    motifs = read_meme(result)
    for m in motifs:
        # STREME ids look like '1-GGACUA'; the part after the dash is its consensus
        m["consensus"] = m["id"].split("-", 1)[1] if "-" in m["id"] else m["id"]
    return motifs


def run_tomtom(query_meme: Path, db_meme: Path, out_dir: Path, thresh: float = 0.1) -> pd.DataFrame:
    """
    Compare query motifs with a database (q-value threshold). Returns the best match per query with
    the database motif's alternative name (usually the protein), or raises PipelineError.
    """
    exe = meme_tool("tomtom")
    if not exe:
        raise PipelineError("tomtom not found on PATH (install MEME Suite: conda install -c bioconda meme)")
    cmd = [exe, "-no-ssc", "-oc", str(out_dir), "-verbosity", "1", "-min-overlap", "5", "-dist", "pearson",
           "-thresh", str(thresh), "-norc", str(query_meme), str(db_meme)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise PipelineError(f"Tomtom failed:\n{e.stderr[-1000:]}") from e
    return parse_tomtom(out_dir / "tomtom.tsv", db_meme)


def parse_tomtom(tsv: Path, db_meme: Optional[Path] = None) -> pd.DataFrame:
    """Best (lowest q-value) Tomtom match per query, with the target's alternative name if known."""
    cols = ["query", "target", "target_name", "p_value", "e_value", "q_value", "overlap", "orientation"]
    if not Path(tsv).exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(tsv, sep="\t", comment="#")
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df.rename(columns={"Query_ID": "query", "Target_ID": "target", "p-value": "p_value",
                            "E-value": "e_value", "q-value": "q_value", "Overlap": "overlap",
                            "Orientation": "orientation"})
    names = {m["id"]: m["alt"] for m in read_meme(db_meme)} if db_meme and Path(db_meme).exists() else {}
    df["target_name"] = [names.get(t, "") for t in df["target"]]
    best = df.sort_values("q_value").groupby("query", as_index=False).first()
    return best[cols].sort_values("q_value").reset_index(drop=True)
