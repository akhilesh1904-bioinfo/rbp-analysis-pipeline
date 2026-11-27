"""YAML config loading, overrides, validation, CLI, and the legacy flag interface."""
from pathlib import Path

import pytest
import yaml

from conftest import write_config
from rbp_pipeline.cli import config_from_legacy_args, legacy_parser, main
from rbp_pipeline.config import (Config, ConfigError, default_yaml, from_dict, load_config,
                                 parse_overrides, validate)


@pytest.fixture
def files(tmp_path):
    for n in ("peaks.bed", "genome.fa", "genes.bed", "a.gtf", "neg.bed"):
        (tmp_path / n).write_text("chr1\t0\t10\n")
    return tmp_path


def minimal(files, **extra):
    data = {"name": "P", "input": "peaks.bed", "reference": {"genome": "genome.fa", "genes_bed": "genes.bed"}}
    data.update(extra)
    return write_config(files / "c.yaml", data)


def test_default_config_is_valid_yaml_with_every_section():
    data = yaml.safe_load(default_yaml())
    cfg = from_dict(data)
    assert cfg.negatives.strategy == "gc-matched"
    assert cfg.peaks.window == 100
    assert set(data) >= {"reference", "peaks", "negatives", "split", "model", "steps"}


def test_relative_paths_resolve_against_config_folder(files):
    cfg = load_config(minimal(files))
    assert Path(cfg.input) == files / "peaks.bed"
    assert Path(cfg.reference.genome) == files / "genome.fa"
    validate(cfg)


def test_home_paths_expand(files, monkeypatch):
    monkeypatch.setenv("HOME", str(files))
    cfg = load_config(minimal(files, input="~/peaks.bed"))
    assert Path(cfg.input) == files / "peaks.bed"


def test_overrides_are_typed():
    o = parse_overrides(["model.max_len=100", "split.test_chroms=[chr1, chr2]", "peaks.window=null",
                         "negatives.strategy=shuffle"])
    assert o == {"model": {"max_len": 100}, "split": {"test_chroms": ["chr1", "chr2"]},
                 "peaks": {"window": None}, "negatives": {"strategy": "shuffle"}}


def test_overrides_applied_on_load(files):
    cfg = load_config(minimal(files), ["model.epochs=3", "negatives.strategy=shuffle"])
    assert cfg.model.epochs == 3 and cfg.negatives.strategy == "shuffle"


def test_unknown_keys_rejected(files):
    with pytest.raises(ConfigError, match="model.max_lenn"):
        load_config(minimal(files, model={"max_lenn": 5}))


def test_validation_reports_all_problems_at_once(tmp_path):
    cfg = Config(name="bad name!", input=str(tmp_path / "missing.bed"))
    cfg.negatives.strategy = "random"
    cfg.split.val_chroms = ["chr21"]
    cfg.peaks.window = 3
    with pytest.raises(ConfigError) as e:
        validate(cfg)
    msg = str(e.value)
    for fragment in ("name 'bad name!'", "input not found", "reference.genome is required",
                     "negatives.strategy", "both split.test_chroms and split.val_chroms", "peaks.window"):
        assert fragment in msg, fragment


def test_split_chroms_must_be_chr_prefixed(files):
    cfg = load_config(minimal(files, split={"test_chroms": ["21"], "val_chroms": ["chr19"]}))
    with pytest.raises(ConfigError, match="chr-prefixed"):
        validate(cfg)


def test_genes_bed_not_needed_with_precomputed_negatives(files):
    cfg = load_config(write_config(files / "c.yaml", {
        "name": "P", "input": "peaks.bed", "reference": {"genome": "genome.fa"},
        "negatives": {"bed": "neg.bed"}}))
    validate(cfg)


def test_legacy_flags_map_to_config():
    args = legacy_parser().parse_args([
        "--input", "x.bed", "--name", "P", "--genome", "g.fa", "--genes-bed", "genes.bed",
        "--window", "100", "--negative-strategy", "shuffle", "--max-len", "80", "--skip-report",
        "--test-chroms", "chr1", "--seed", "7"])
    cfg = config_from_legacy_args(args)
    assert (cfg.peaks.window, cfg.negatives.strategy, cfg.model.max_len, cfg.seed) == (100, "shuffle", 80, 7)
    assert cfg.split.test_chroms == ["chr1"] and cfg.steps.report is False and cfg.steps.annotation is True


def test_cli_default_config(capsys):
    assert main(["default-config"]) == 0
    assert "negatives:" in capsys.readouterr().out


def test_cli_validate(files, capsys):
    assert main(["validate", str(minimal(files))]) == 0
    assert "OK" in capsys.readouterr().out
    assert main(["validate", str(minimal(files, input="nope.bed"))]) == 1
    assert "input not found" in capsys.readouterr().err
