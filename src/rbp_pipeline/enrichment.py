"""
Gene-set enrichment of bound genes with g:Profiler (https://biit.cs.ut.ee/gprofiler), against an
explicit background gene universe.

Background choice matters: genes containing eCLIP peaks are long and highly expressed, so testing
them against every annotated gene mostly finds "expressed in this cell line". Options:
  * background_genes file (best): genes expressed in the eCLIP cell line (e.g. K562 / HepG2 RNA-seq)
  * sampled (default): genes overlapped by any peak or any negative region; negatives are drawn
    from the same region set, so this is the universe the peaks could have come from
  * annotated: all genes g:Profiler knows about (inflated; only for comparison)

A custom background is sent with domain_scope "custom_annotated", so genes without any annotation
in a source (many lncRNAs and pseudogenes) don't dilute the background, and by default query and
background are restricted to protein-coding genes. The effective background size g:Profiler used
is logged and checked against the list that was sent.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd

log = logging.getLogger("rbp_pipeline")

GPROFILER_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
COLUMNS = ["source", "native", "name", "p_value", "term_size", "query_size", "intersection_size",
           "effective_domain_size", "precision", "recall"]


def gene_list(names: Iterable[str]) -> List[str]:
    """Unique, sorted gene symbols with placeholders removed."""
    return sorted({str(n) for n in names if n and str(n) not in (".", "nan", "None")})


def read_gene_file(path: Path) -> List[str]:
    return gene_list(line.strip().split("\t")[0] for line in Path(path).read_text().splitlines()
                     if line.strip() and not line.startswith("#"))


def gprofiler(query: Sequence[str], background: Optional[Sequence[str]], organism: str = "hsapiens",
              sources: Sequence[str] = ("GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"),
              threshold: float = 0.05, timeout: int = 120) -> pd.DataFrame:
    """
    Run g:GOSt (g:SCS multiple-testing correction). `background=None` uses all annotated genes;
    a custom background is applied over annotated genes only (domain_scope "custom_annotated").
    Raises urllib/OSError/ValueError on network or API problems.
    """
    payload: Dict = {"organism": organism, "query": list(query), "sources": list(sources),
                     "user_threshold": threshold, "significance_threshold_method": "g_SCS",
                     "no_evidences": True, "all_results": False}
    if background:
        payload.update(domain_scope="custom_annotated", background=list(background))
    else:
        payload["domain_scope"] = "annotated"
    req = urllib.request.Request(GPROFILER_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "rbp-pipeline"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    rows = data.get("result") or []
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    return df[[c for c in COLUMNS if c in df.columns]].sort_values("p_value").reset_index(drop=True)


def run_enrichment(bound_genes: Sequence[str], universe: Optional[Sequence[str]], out_dir: Path, *,
                   organism: str, sources: Sequence[str], restrict_to: Optional[Set[str]] = None) -> Dict:
    """
    Write query/background lists and results; never raises (enrichment is optional).
    restrict_to: keep only these genes (e.g. protein-coding) in the query and the background.
    """
    query = gene_list(bound_genes)
    if restrict_to is not None:
        query = [g for g in query if g in restrict_to]
        if universe is not None:
            universe = [g for g in gene_list(universe) if g in restrict_to]
    (out_dir / "enrichment_query_genes.txt").write_text("\n".join(query) + "\n")
    if universe is not None:
        universe = gene_list(set(universe) | set(query))   # the query must be inside the background
        (out_dir / "enrichment_background_genes.txt").write_text("\n".join(universe) + "\n")
    summary = {"n_query": len(query), "n_background": len(universe) if universe is not None else "annotated"}
    if not query:
        log.info("  Enrichment skipped: no bound genes (is reference.gtf set?)")
        return {**summary, "status": "skipped: no genes"}
    try:
        df = gprofiler(query, universe, organism, sources)
    except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
        log.warning(f"  Warning: g:Profiler request failed ({e}); enrichment skipped. "
                    "It needs internet access at run time.")
        return {**summary, "status": f"failed: {e}"}
    df.to_csv(out_dir / "enrichment_gprofiler.csv", index=False)
    domain = effective_domain(df)
    log.info(f"  g:Profiler          : {len(df):,} significant terms "
             f"({len(query):,} bound genes vs {summary['n_background']} background"
             + (f"; effective background {domain:,}" if domain else "") + ")")
    warning = background_warning(domain, universe)
    if warning:
        log.warning(f"  WARNING: {warning}")
    for r in df.head(5).itertuples():
        log.info(f"    {r.source:<6} {r.name[:55]:<55} p = {r.p_value:.2e}")
    return {**summary, "status": "ok", "n_terms": int(len(df)), "effective_domain_size": domain,
            "warning": warning,
            "top_terms": [{"source": r.source, "name": r.name, "p_value": float(r.p_value)}
                          for r in df.head(10).itertuples()]}


def effective_domain(df: pd.DataFrame) -> Optional[int]:
    """Largest background size g:Profiler reports across sources (None if unavailable)."""
    if "effective_domain_size" not in df or df.empty:
        return None
    return int(pd.to_numeric(df["effective_domain_size"], errors="coerce").max())


def background_warning(domain: Optional[int], universe: Optional[Sequence[str]]) -> Optional[str]:
    """Flag a custom background that g:Profiler didn't apply (domain far larger than the list sent)."""
    if domain is None or universe is None or not len(universe):
        return None
    if domain > 1.2 * len(universe):
        return (f"g:Profiler used a background of {domain:,} genes but {len(universe):,} were sent; the custom "
                "background may not have been applied (check gene symbols and organism).")
    return None
