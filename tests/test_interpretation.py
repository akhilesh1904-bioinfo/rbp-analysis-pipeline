"""Phase C: model interpretation, motifs and enrichment."""
import io
import json
import os
import random
import stat
import sys
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import MOTIF, needs_tools
from rbp_pipeline import enrichment
from rbp_pipeline.attribution import (cnn_importance, consensus, filter_motifs, information_content, ism,
                                      ism_importance, kmer_importance, lr_kmer_table, position_profile,
                                      saliency)
from rbp_pipeline.config import ConfigError, load_config, validate
from rbp_pipeline.dataset import one_hot
from rbp_pipeline.models import RBP_CNN, set_seed, train_lr
from rbp_pipeline.motifs import (bh_adjust, iupac_matches, iupac_regex, known_motif_enrichment,
                                 parse_tomtom, read_meme, run_streme, run_tomtom, write_meme)
from rbp_pipeline.pipeline import run

CPU = torch.device("cpu")


def random_seq(rng, n):
    return "".join(rng.choice("ACGT") for _ in range(n))


def planted(rng, n, motif=MOTIF, length=60):
    out = []
    for _ in range(n):
        s = random_seq(rng, length)
        p = rng.randint(5, length - len(motif) - 5)
        out.append(s[:p] + motif + s[p + len(motif):])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CNN attributions
# ─────────────────────────────────────────────────────────────────────────────

def test_ism_matches_brute_force():
    set_seed(0)
    model = RBP_CNN().eval()
    seq = "ACGTTGCANNGGCATTACG"
    delta = ism(model, [seq], max_len=24, device=CPU)[0]
    assert delta.shape == (len(seq), 4)

    def logit(s):
        with torch.no_grad():
            return model(torch.from_numpy(one_hot(s, 24))[None]).item()
    ref = logit(seq)
    for i in (0, 5, 18):
        for j, b in enumerate("ACGT"):
            mutant = seq[:i] + b + seq[i + 1:]
            expected = 0.0 if b == seq[i] else logit(mutant) - ref
            assert delta[i, j] == pytest.approx(expected, abs=1e-5)
    assert not delta[8].any() and not delta[9].any()          # N positions untouched
    imp = ism_importance(delta, seq)
    assert imp[5] == pytest.approx(-delta[5].sum() / 3, abs=1e-6)


def test_saliency_and_dispatch_shapes():
    set_seed(0)
    model = RBP_CNN().eval()
    seqs = ["ACGT" * 10, "GGAC" * 40]                        # second one is centre-cropped to max_len
    sal = saliency(model, seqs, max_len=50, device=CPU)
    assert [len(s) for s in sal] == [40, 50]
    cropped, imps = cnn_importance(model, seqs, 50, "ism", CPU)
    assert [len(s) for s in cropped] == [40, 50] and [len(i) for i in imps] == [40, 50]
    with pytest.raises(ValueError):
        cnn_importance(model, seqs, 50, "nope", CPU)


def test_kmer_importance_finds_the_important_kmer():
    rng = random.Random(0)
    seqs, imps = [], []
    for s in planted(rng, 60):
        p = s.index(MOTIF)
        imp = np.zeros(len(s), dtype=np.float32)
        imp[p:p + len(MOTIF)] = 1.0
        seqs.append(s); imps.append(imp)
    df = kmer_importance(seqs, imps, k=5)
    assert df.iloc[0]["kmer"] in MOTIF
    assert df.iloc[0]["mean_importance"] > 4
    assert df.iloc[0]["kmer_rna"] == df.iloc[0]["kmer"].replace("T", "U")
    prof = position_profile(imps)
    assert {"offset_from_centre", "mean_importance", "sd", "n"} <= set(prof.columns)
    assert prof["n"].max() == 60


def test_filter_motifs_recover_a_hand_built_filter():
    """Filter 0 is a one-hot detector for TGCATGT; its motif and influence must come out on top."""
    model = RBP_CNN().eval()
    with torch.no_grad():
        model.conv1.weight.zero_(); model.conv1.bias.fill_(-6.0)
        for j, b in enumerate("TGCATGT"):
            model.conv1.weight[0, "ACGT".index(b), j] = 1.0
        model.conv2.weight.zero_(); model.conv2.weight[:, 0, :] = 1.0
        model.conv2.bias.zero_()
        for layer in model.classifier:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.fill_(0.1); layer.bias.zero_()
    rng = random.Random(1)
    seqs = planted(rng, 40, motif="TGCATGT")
    motifs = filter_motifs(model, seqs, max_len=60, device=CPU, min_sites=5)
    assert motifs and motifs[0]["filter"] == 1
    assert motifs[0]["consensus"] == "UGCAUGU"
    assert motifs[0]["influence"] > 0
    assert information_content(motifs[0]["pwm"]).sum() > 10


def test_consensus_lowercases_uninformative_positions():
    pwm = np.array([[1, 0, 0, 0], [0.25, 0.25, 0.25, 0.25], [0, 0, 0, 1]], dtype=float)
    assert consensus(pwm) == "AaU"


def test_lr_kmer_table_ranks_motif_kmers_first():
    rng = random.Random(2)
    pos = planted(rng, 150, motif="GGACGGAC")
    neg = [random_seq(rng, 60) for _ in range(150)]
    clf, vec = train_lr(pos + neg, [1] * 150 + [0] * 150, seed=0, ks=(3, 4))
    df = lr_kmer_table(clf, vec, pos + neg, [1] * 150 + [0] * 150)
    assert set(df["k"]) == {3, 4}
    assert df.iloc[0]["kmer"] in "GGACGGAC"
    assert df.iloc[0]["mean_count_pos"] > df.iloc[0]["mean_count_neg"]
    assert df["coefficient"].is_monotonic_decreasing


def test_lr_kmer_sizes_include_single_bases():
    clf, vec = train_lr(["ACGTACGT", "GGGGCCCC"] * 5, [1, 0] * 5, seed=0, ks=(1, 2))
    names = set(vec.get_feature_names_out())
    assert {"A", "C", "G", "T"} <= names and "AC" in names


# ─────────────────────────────────────────────────────────────────────────────
# Known motifs and MEME format
# ─────────────────────────────────────────────────────────────────────────────

def test_iupac():
    assert iupac_regex("DRACH").findall("GGACT") == ["GGACT"]
    assert len(iupac_regex("UGUG").findall("TGTGTG")) == 2           # overlapping
    assert iupac_matches("GGAC", "AGGACU") and not iupac_matches("GGAC", "GGTC")
    assert iupac_matches("CAUH", "CATA") and not iupac_matches("CAUH", "CATG")


def test_known_motif_enrichment():
    rng = random.Random(3)
    pos = planted(rng, 200, motif="GGACT")
    neg = [random_seq(rng, 60) for _ in range(200)]
    df = known_motif_enrichment(pos, neg, {"m6A": "DRACH", "poly_C": "CCCCCCCC"})
    top = df.iloc[0]
    assert top["motif"] == "m6A" and top["frac_pos"] == 1.0 and top["odds_ratio"] > 1
    assert top["q_value"] < 0.01
    assert top["iupac"] == "DRACH"


def test_bh_adjust():
    q = bh_adjust([0.01, 0.04, 0.03, 0.5])
    assert q == pytest.approx([0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])
    assert bh_adjust([]).size == 0


def test_meme_roundtrip(tmp_path):
    pwm = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.25] * 4])
    path = write_meme([{"id": "m1", "alt": "GGN", "pwm": pwm, "n_sites": 33}], tmp_path / "x.meme")
    back = read_meme(path)
    assert len(back) == 1 and back[0]["id"] == "m1" and back[0]["alt"] == "GGN"
    assert back[0]["nsites"] == 33 and np.allclose(back[0]["pwm"], pwm)


STREME_TXT = """MEME version 5.5.5

ALPHABET= ACGU

strands: +

Background letter frequencies
A 0.25 C 0.25 G 0.25 U 0.25

MOTIF 1-UGCAUGU STREME-1
letter-probability matrix: alength= 4 w= 7 nsites= 412 E= 2.3e-045
 0.000 0.000 0.000 1.000
 0.000 0.000 1.000 0.000
 0.000 1.000 0.000 0.000
 1.000 0.000 0.000 0.000
 0.000 0.000 0.000 1.000
 0.000 0.000 1.000 0.000
 0.000 0.000 0.000 1.000

MOTIF 2-AAGG STREME-2
letter-probability matrix: alength= 4 w= 4 nsites= 50 E= 3.1e-001
 0.900 0.033 0.033 0.034
 0.900 0.033 0.033 0.034
 0.033 0.033 0.900 0.034
 0.033 0.033 0.900 0.034
"""

TOMTOM_TSV = """Query_ID\tTarget_ID\tOptimal_offset\tp-value\tE-value\tq-value\tOverlap\tQuery_consensus\tTarget_consensus\tOrientation
1-UGCAUGU\tRNCMPT00103\t0\t1e-06\t1e-04\t2e-04\t7\tUGCAUGU\tUGCAUGU\t+
1-UGCAUGU\tRNCMPT00999\t1\t1e-03\t1e-01\t5e-02\t6\tUGCAUGU\tGCAUGA\t+
filter_3\tRNCMPT00103\t0\t1e-04\t1e-02\t1e-02\t7\tUGCAUGG\tUGCAUGU\t+

# Tomtom (Motif Comparison Tool): Version 5.5.5
"""

DB_MEME = """MEME version 4

ALPHABET= ACGU

MOTIF RNCMPT00103 RBFOX1
letter-probability matrix: alength= 4 w= 7 nsites= 20 E= 0
0 0 0 1
0 0 1 0
0 1 0 0
1 0 0 0
0 0 0 1
0 0 1 0
0 0 0 1

MOTIF RNCMPT00999 OTHER
letter-probability matrix: alength= 4 w= 2 nsites= 20 E= 0
1 0 0 0
0 1 0 0
"""


@pytest.fixture
def fake_meme(tmp_path, monkeypatch):
    """Fake `streme` and `tomtom` on PATH that write recorded MEME Suite outputs and log their argv."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (tmp_path / "streme.txt.fixture").write_text(STREME_TXT)
    (tmp_path / "tomtom.tsv.fixture").write_text(TOMTOM_TSV)
    for tool, flag, out_name in (("streme", "--oc", "streme.txt"), ("tomtom", "-oc", "tomtom.tsv")):
        script = bin_dir / tool
        script.write_text(f"""#!{sys.executable}
import sys, pathlib, json
args = sys.argv[1:]
out = pathlib.Path(args[args.index("{flag}") + 1]); out.mkdir(parents=True, exist_ok=True)
(out / "{out_name}").write_text(pathlib.Path({str(tmp_path / (out_name + '.fixture'))!r}).read_text())
pathlib.Path({str(tmp_path / (tool + '.argv'))!r}).write_text(json.dumps(args))
""")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    db = tmp_path / "db.meme"
    db.write_text(DB_MEME)
    return {"db": db, "tmp": tmp_path}


def test_streme_wrapper(fake_meme, tmp_path):
    motifs = run_streme(["ACGT" * 10], ["TTTT" * 10], tmp_path / "out", nmotifs=3)
    assert [m["consensus"] for m in motifs] == ["UGCAUGU", "AAGG"]
    assert motifs[0]["evalue"] == pytest.approx(2.3e-45) and motifs[0]["nsites"] == 412
    args = json.loads((fake_meme["tmp"] / "streme.argv").read_text())
    assert "--rna" in args and args[args.index("--nmotifs") + 1] == "3"
    fa = (tmp_path / "out" / "streme_positives.fa").read_text()
    assert "U" in fa and "T" not in fa.replace(">", "").split("\n", 1)[1]    # RNA letters


def test_tomtom_wrapper_and_parser(fake_meme, tmp_path):
    q = write_meme([{"id": "1-UGCAUGU", "pwm": np.eye(4)[[3, 2, 1, 0]]}], tmp_path / "q.meme")
    best = run_tomtom(q, fake_meme["db"], tmp_path / "tt")
    assert list(best["query"]) == ["1-UGCAUGU", "filter_3"]
    assert best.iloc[0]["target"] == "RNCMPT00103" and best.iloc[0]["target_name"] == "RBFOX1"
    assert "-norc" in json.loads((fake_meme["tmp"] / "tomtom.argv").read_text())
    assert parse_tomtom(tmp_path / "missing.tsv").empty


def test_missing_meme_tools_raise(monkeypatch, tmp_path):
    from rbp_pipeline import PipelineError
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(PipelineError, match="streme not found"):
        run_streme(["A"], ["C"], tmp_path, 1)


# ─────────────────────────────────────────────────────────────────────────────
# g:Profiler
# ─────────────────────────────────────────────────────────────────────────────

class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_gprofiler_request_uses_custom_background(monkeypatch, tmp_path):
    sent = {}

    def fake_urlopen(req, timeout=0):
        sent.update(json.loads(req.data.decode()))
        return _Resp(json.dumps({"result": [
            {"source": "GO:BP", "native": "GO:1", "name": "mRNA stability", "p_value": 1e-5, "term_size": 50,
             "query_size": 3, "intersection_size": 3, "effective_domain_size": 10, "precision": 1, "recall": 0.06,
             "parents": []}]}).encode())
    monkeypatch.setattr(enrichment.urllib.request, "urlopen", fake_urlopen)

    out = enrichment.run_enrichment(["MYC", "IGF2", ".", "MYC"], ["ACTB", "GAPDH"], tmp_path,
                                    organism="hsapiens", sources=["GO:BP"])
    assert sent["domain_scope"] == "custom" and sent["query"] == ["IGF2", "MYC"]
    assert set(sent["background"]) == {"ACTB", "GAPDH", "IGF2", "MYC"}        # query added to background
    assert out["status"] == "ok" and out["n_terms"] == 1
    assert out["top_terms"][0]["name"] == "mRNA stability"
    assert pd.read_csv(tmp_path / "enrichment_gprofiler.csv").iloc[0]["native"] == "GO:1"
    assert (tmp_path / "enrichment_background_genes.txt").read_text().split() == ["ACTB", "GAPDH", "IGF2", "MYC"]


def test_gprofiler_failure_does_not_crash(monkeypatch, tmp_path):
    def boom(req, timeout=0):
        raise urllib.error.URLError("no network")
    monkeypatch.setattr(enrichment.urllib.request, "urlopen", boom)
    out = enrichment.run_enrichment(["MYC"], None, tmp_path, organism="hsapiens", sources=["GO:BP"])
    assert out["status"].startswith("failed")
    out = enrichment.run_enrichment(["."], None, tmp_path, organism="hsapiens", sources=["GO:BP"])
    assert out["status"].startswith("skipped")


def test_read_gene_file(tmp_path):
    f = tmp_path / "g.txt"
    f.write_text("# expressed\nMYC\tx\nACTB\n\nMYC\n")
    assert enrichment.read_gene_file(f) == ["ACTB", "MYC"]


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def test_interpretation_config_validation(synthetic_config):
    cfg_path, d = synthetic_config
    validate(load_config(cfg_path, ["interpretation.known_motifs={X: GGAC}", "model.lr_kmers=[3, 4, 5]"]))
    bad = load_config(cfg_path, ["interpretation.method=shap", "interpretation.kmer_size=2",
                                 "interpretation.known_motifs={X: GGZZ}", "interpretation.background=all",
                                 "interpretation.motif_db=missing.meme", "model.lr_kmers=[0]"])
    with pytest.raises(ConfigError) as e:
        validate(bad)
    msg = str(e.value)
    for key in ("interpretation.method", "interpretation.kmer_size", "known_motifs.X", "interpretation.background",
                "interpretation.motif_db not found", "model.lr_kmers"):
        assert key in msg, key
    # relative motif_db paths are resolved against the config's folder
    (d["tmp"] / "db.meme").write_text(DB_MEME)
    assert load_config(cfg_path, ["interpretation.motif_db=db.meme"]).interpretation.motif_db == str(d["tmp"] / "db.meme")


# ─────────────────────────────────────────────────────────────────────────────
# End to end with (fake) MEME Suite and (mocked) g:Profiler
# ─────────────────────────────────────────────────────────────────────────────

@needs_tools
def test_pipeline_interpretation_with_motif_tools(synthetic_config, fake_meme, monkeypatch):
    cfg_path, d = synthetic_config
    captured = {}

    def fake_gprofiler(query, background, organism, sources, **kw):
        captured.update(query=query, background=background)
        return pd.DataFrame([{"source": "GO:BP", "native": "GO:1", "name": "x", "p_value": 0.01}])
    monkeypatch.setattr(enrichment, "gprofiler", fake_gprofiler)

    res = run(load_config(cfg_path, [f"interpretation.motif_db={fake_meme['db']}", "interpretation.enrichment=true",
                                     "interpretation.n_sequences=40", "interpretation.method=saliency",
                                     "model.lr_kmers=[3, 4]"]))
    out = d["tmp"] / "results" / "SYN"
    s = res["interpretation"]
    assert s["streme"]["status"] == "ok" and s["streme"]["motifs"][0]["consensus"] == "UGCAUGU"
    assert s["tomtom"]["status"] == "ok"
    tt = pd.read_csv(out / "interpretation" / "tomtom_best_matches.csv")
    assert {"query", "query_source", "query_consensus", "target_name"} <= set(tt.columns)
    assert s["enrichment"]["status"] == "ok" and s["enrichment"]["background"].startswith("genes overlapping")
    assert "GENEA" in captured["query"] and set(captured["query"]) <= set(captured["background"])
    assert s["cnn"]["n_sequences"] == 40 and s["method"] == "saliency"
    assert set(s["lr"]["kmer_sizes"]) == {3, 4}
    for fig in ("interp_streme_motifs", "interp_lr_kmers", "interp_position_importance"):
        assert (out / "figures" / f"{fig}.png").exists(), fig
    assert json.loads((out / "SYN_metrics.json").read_text())["interpretation"]["streme"]["status"] == "ok"
