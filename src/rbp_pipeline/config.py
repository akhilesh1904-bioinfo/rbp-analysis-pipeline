"""Run configuration: YAML file -> validated, typed Config object."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from . import PipelineError


class ConfigError(PipelineError):
    """Invalid or inconsistent configuration."""


@dataclass
class ReferenceConfig:
    genome: Optional[str] = None      # GRCh38 FASTA with chr-prefixed names
    genes_bed: Optional[str] = None   # regions negatives are drawn from (genes, exons, introns...)
    gtf: Optional[str] = None         # optional GENCODE GTF for region annotation


@dataclass
class PeaksConfig:
    min_log2fc: Optional[float] = None   # ENCODE recommends 3
    min_log10p: Optional[float] = None   # ENCODE recommends 3
    window: Optional[int] = None         # resize peaks to a fixed width around the midpoint


@dataclass
class NegativesConfig:
    strategy: str = "gc-matched"         # gc-matched | shuffle
    gc_candidates: int = 10
    bed: Optional[str] = None            # pre-computed negatives; skips generation


@dataclass
class SplitConfig:
    test_chroms: List[str] = field(default_factory=lambda: ["chr21", "chr22", "chrX"])
    val_chroms: List[str] = field(default_factory=lambda: ["chr19", "chr20"])


@dataclass
class ModelConfig:
    max_len: int = 126
    epochs: int = 30
    patience: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3


@dataclass
class StepsConfig:
    annotation: bool = True
    report: bool = True


@dataclass
class Config:
    name: str = ""
    input: str = ""
    output_dir: str = "results"
    seed: int = 42
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    peaks: PeaksConfig = field(default_factory=PeaksConfig)
    negatives: NegativesConfig = field(default_factory=NegativesConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    steps: StepsConfig = field(default_factory=StepsConfig)

    @property
    def out_dir(self) -> Path:
        return Path(self.output_dir) / self.name

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


NEGATIVE_STRATEGIES = ("gc-matched", "shuffle")
INPUT_PATH_FIELDS = ("input", "reference.genome", "reference.genes_bed", "reference.gtf", "negatives.bed")


# ─────────────────────────────────────────────────────────────────────────────
# Building a Config from dictionaries / YAML
# ─────────────────────────────────────────────────────────────────────────────

def _build(cls, data: Dict[str, Any], prefix: str = ""):
    if not isinstance(data, dict):
        raise ConfigError(f"'{prefix.rstrip('.') or 'config'}' must be a mapping, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(f"Unknown config key(s): {', '.join(prefix + k for k in unknown)}")
    kwargs = {}
    defaults = cls()
    for name, value in data.items():
        default = getattr(defaults, name)
        if is_dataclass(default):
            kwargs[name] = _build(type(default), value or {}, prefix + name + ".")
        else:
            kwargs[name] = value
    return cls(**kwargs)


def from_dict(data: Dict[str, Any]) -> Config:
    return _build(Config, data or {})


def _set_dotted(data: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = data
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise ConfigError(f"Cannot set '{dotted}': '{k}' is not a section")
    node[keys[-1]] = value


def _get_dotted(cfg: Config, dotted: str):
    obj = cfg
    for k in dotted.split("."):
        obj = getattr(obj, k)
    return obj


def _set_attr_dotted(cfg: Config, dotted: str, value) -> None:
    keys = dotted.split(".")
    obj = cfg
    for k in keys[:-1]:
        obj = getattr(obj, k)
    setattr(obj, keys[-1], value)


def parse_overrides(overrides: Sequence[str]) -> Dict[str, Any]:
    """Turn ['model.max_len=100', 'split.test_chroms=[chr1]'] into a nested dict (values parsed as YAML)."""
    out: Dict[str, Any] = {}
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"Override '{item}' must look like key.subkey=value")
        key, raw = item.split("=", 1)
        _set_dotted(out, key.strip(), yaml.safe_load(raw) if raw.strip() else None)
    return out


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(path: Path, overrides: Sequence[str] = ()) -> Config:
    """
    Load a YAML config. Input file paths that are relative are resolved against the
    config file's folder, so a config and its data can be moved together.
    `output_dir` is resolved against the current working directory.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Could not parse {path}: {e}") from e
    cfg = from_dict(_deep_merge(data, parse_overrides(overrides)))
    base = path.resolve().parent
    for dotted in INPUT_PATH_FIELDS:
        value = _get_dotted(cfg, dotted)
        if value and not Path(value).expanduser().is_absolute():
            _set_attr_dotted(cfg, dotted, str(base / value))
        elif value:
            _set_attr_dotted(cfg, dotted, str(Path(value).expanduser()))
    return cfg


def default_yaml() -> str:
    """A fully commented default config, for `rbp-pipeline default-config`."""
    return DEFAULT_CONFIG_TEXT


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate(cfg: Config) -> None:
    """Check every setting and input file before any work starts; report all problems at once."""
    problems: List[str] = []

    def need_file(label: str, value: Optional[str], required: bool = True):
        if not value:
            if required:
                problems.append(f"{label} is required")
            return
        if not Path(value).is_file():
            problems.append(f"{label} not found: {value}")

    if not cfg.name:
        problems.append("name is required (e.g. the protein name)")
    elif not re.fullmatch(r"[A-Za-z0-9._-]+", cfg.name):
        problems.append(f"name '{cfg.name}' may only contain letters, digits, '.', '_' and '-'")

    need_file("input", cfg.input)
    need_file("reference.genome", cfg.reference.genome)
    need_file("reference.gtf", cfg.reference.gtf, required=False)
    if cfg.negatives.bed:
        need_file("negatives.bed", cfg.negatives.bed)
    else:
        need_file("reference.genes_bed (needed to generate negatives)", cfg.reference.genes_bed)

    if cfg.negatives.strategy not in NEGATIVE_STRATEGIES:
        problems.append(f"negatives.strategy must be one of {NEGATIVE_STRATEGIES}, got '{cfg.negatives.strategy}'")
    if not isinstance(cfg.negatives.gc_candidates, int) or cfg.negatives.gc_candidates < 1:
        problems.append("negatives.gc_candidates must be a whole number >= 1")
    if cfg.peaks.window is not None and (not isinstance(cfg.peaks.window, int) or cfg.peaks.window < 8):
        problems.append("peaks.window must be a whole number >= 8 (or null)")
    for key in ("min_log2fc", "min_log10p"):
        v = getattr(cfg.peaks, key)
        if v is not None and not isinstance(v, (int, float)):
            problems.append(f"peaks.{key} must be a number (or null)")

    test, val = set(cfg.split.test_chroms or []), set(cfg.split.val_chroms or [])
    if not test:
        problems.append("split.test_chroms must list at least one chromosome")
    if not val:
        problems.append("split.val_chroms must list at least one chromosome")
    if test & val:
        problems.append(f"chromosomes in both split.test_chroms and split.val_chroms: {sorted(test & val)}")
    bad = sorted(c for c in test | val if not str(c).startswith("chr"))
    if bad:
        problems.append(f"split chromosomes must use chr-prefixed names (e.g. chr21), got {bad}")

    m = cfg.model
    for key, minimum in (("max_len", 8), ("epochs", 1), ("patience", 1), ("batch_size", 1)):
        v = getattr(m, key)
        if not isinstance(v, int) or v < minimum:
            problems.append(f"model.{key} must be a whole number >= {minimum}")
    if not isinstance(m.learning_rate, (int, float)) or m.learning_rate <= 0:
        problems.append("model.learning_rate must be a positive number")
    if not isinstance(cfg.seed, int):
        problems.append("seed must be a whole number")

    if problems:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(problems))


DEFAULT_CONFIG_TEXT = """\
# rbp-pipeline configuration
# Relative input paths are resolved against this file's folder.
# Any value can be overridden on the command line, e.g. --set model.max_len=100

name: MY_PROTEIN                 # used for the output folder and file names
input: peaks.bed                 # ENCODE eCLIP peak BED (chr1 or 1 naming both accepted)
output_dir: results              # results go to <output_dir>/<name>/ (relative to where you run)
seed: 42

reference:
  genome: GRCh38.primary_assembly.genome.fa   # chr-prefixed FASTA (GENCODE/UCSC style)
  genes_bed: genes_only.bed      # regions negatives are drawn from (genes, exons or introns)
  gtf: null                      # optional GENCODE GTF for exonic/intronic/UTR annotation

peaks:
  min_log2fc: null               # keep peaks with log2 fold-enrichment (BED col 7) >= this; ENCODE uses 3
  min_log10p: null               # keep peaks with -log10 p (BED col 8) >= this; ENCODE uses 3
  window: 100                    # resize every peak to this width around its midpoint; null = as given

negatives:
  strategy: gc-matched           # gc-matched | shuffle
  gc_candidates: 10              # candidate regions per peak for gc-matched
  bed: null                      # pre-computed negatives BED (skips generation)

split:
  test_chroms: [chr21, chr22, chrX]
  val_chroms: [chr19, chr20]

model:
  max_len: 126                   # CNN input length; longer sequences are centre-cropped
  epochs: 30
  patience: 5                    # early-stopping patience (epochs without val-loss improvement)
  batch_size: 64
  learning_rate: 0.001

steps:
  annotation: true               # needs reference.gtf
  report: true                   # Word report (needs Node.js)
"""
