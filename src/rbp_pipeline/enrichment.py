"""
Gene-set enrichment of bound genes with g:Profiler (https://biit.cs.ut.ee/gprofiler), against an
explicit background gene universe.

Background choice matters: genes containing eCLIP peaks are long and highly expressed, so testing
them against every annotated gene mostly finds "expressed in this cell line". Options:
  * background_genes file (best): genes expressed in the eCLIP cell line (e.g. K562 / HepG2 RNA-seq)
  * sampled (default): genes overlapped by any peak or any negative region; negatives are drawn
    from the same region set, so this is the universe the peaks could have come from
  * annotated: all genes g:Profiler knows about (inflated; only for comparison)
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

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
    Run g:GOSt (g:SCS multiple-testing correction). `background=None` uses all annotated genes.
    Raises urllib/OSError/ValueError on network or API problems.
    """
    payload: Dict = {"organism": organism, "query": list(query), "sources": list(sources),
                     "user_threshold": threshold, "significance_threshold_method": "g_SCS",
                     "no_evidences": True, "all_results": False}
    if background:
        payload.update(domain_scope="custom", background=list(background))
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
                   organism: str, sources: Sequence[str]) -> Dict:
    """Write query/background lists and results; never raises (enrichment is optional)."""
    query = gene_list(bound_genes)
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
    log.info(f"  g:Profiler          : {len(df):,} significant terms "
             f"({len(query):,} bound genes vs {summary['n_background']} background)")
    for r in df.head(5).itertuples():
        log.info(f"    {r.source:<6} {r.name[:55]:<55} p = {r.p_value:.2e}")
    return {**summary, "status": "ok", "n_terms": int(len(df)),
            "top_terms": [{"source": r.source, "name": r.name, "p_value": float(r.p_value)}
                          for r in df.head(10).itertuples()]}
