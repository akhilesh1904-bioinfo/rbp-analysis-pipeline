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
    assembly: str = "GRCh38"          # GRCh38 | GRCh37 | any (skip the build check)


@dataclass
class PeaksConfig:
    min_log2fc: Optional[float] = None   # ENCODE recommends 3
    min_log10p: Optional[float] = None   # ENCODE recommends 3
    window: Optional[int] = None         # resize peaks to a fixed width around the midpoint


@dataclass
class NegativesConfig:
    strategy: str = "gc-matched"         # gc-matched | shuffle
    gc_candidates: int = 10
    source: str = "genes_bed"            # genes_bed (reference.genes_bed) | gtf (UTR/exonic/intronic sets from reference.gtf)
    region_classes: List[str] = field(default_factory=lambda: ["UTR", "exonic", "intronic"])  # with source: gtf
    match_region: bool = False           # with source: gtf, draw each peak's negatives from its own region class
    strand_aware: bool = True            # place '+' peaks in '+' regions and '-' in '-' (needs stranded regions)
    qc: bool = True                      # compare region/strand of negatives with positives (needs reference.gtf)
    bed: Optional[str] = None            # pre-computed negatives; skips generation


@dataclass
class SplitConfig:
    test_chroms: List[str] = field(default_factory=lambda: ["chr21", "chr22", "chrX"])
    val_chroms: List[str] = field(default_factory=lambda: ["chr19", "chr20"])
    leakage: str = "report"           # report | remove : val/test sequences resembling training ones
    leakage_kmer: int = 20            # shared exact substring length that counts as resemblance


@dataclass
class ModelConfig:
    max_len: int = 126
    epochs: int = 30
    patience: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    dropout: float = 0.3
    weight_decay: float = 0.0         # >0 switches the optimiser to AdamW
    lr_kmers: List[int] = field(default_factory=lambda: [2, 3])   # k-mer sizes for logistic regression


@dataclass
class EvaluationConfig:
    n_seeds: int = 1                  # retrain models with this many seeds; report mean ± SD
    bootstrap: int = 1000             # bootstrap resamples for 95% CIs (0 = off)
    threshold: float = 0.5            # probability cut-off for precision/recall/F1/MCC


# A small panel of RNA sequence elements frequently reported for RBPs (IUPAC, RNA alphabet), checked
# for enrichment in every run as a sanity check. Replace or extend per protein in the config;
# sources are in the README.
DEFAULT_KNOWN_MOTIFS = {
    "GGAC": "GGAC",               # Biswas et al. 2019
    "CAUH": "CAUH",               # Hafner et al. 2010
    "m6A_DRACH": "DRACH",         # m6A consensus (Huang et al. 2018)
    "GGUG": "GGUG",               # Lerga et al. 2001
    "UG_repeat": "UGUGUG",        # Buratti & Baralle 2001
    "ARE_AUUUA": "AUUUA",         # AU-rich element pentamer
}


@dataclass
class InterpretationConfig:
    n_sequences: int = 500            # top-scoring held-out test positives explained by the CNN
    method: str = "ism"               # ism (in-silico mutagenesis) | saliency (gradient x input)
    kmer_size: int = 5                # CNN importance is summed over k-mers of this size
    top_kmers: int = 20               # rows in the CNN / LR k-mer tables
    motifs: bool = True               # STREME + Tomtom (MEME Suite on PATH); skipped with a note otherwise
    streme_nmotifs: int = 8
    motif_db: Optional[str] = None    # MEME-format RNA motif database for Tomtom (e.g. Ray2013_rbp_Homo_sapiens.meme)
    known_motifs: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_KNOWN_MOTIFS))
    enrichment: bool = False          # g:Profiler gene-set enrichment (needs internet and reference.gtf)
    organism: str = "hsapiens"
    enrichment_sources: List[str] = field(default_factory=lambda: ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"])
    background: str = "sampled"       # sampled (genes overlapping peaks or negatives) | annotated (all g:Profiler genes)
    background_genes: Optional[str] = None   # gene-symbol list (one per line) used instead, e.g. expressed genes
    protein_coding_only: bool = True  # restrict bound genes and background to protein-coding genes (needs reference.gtf)


@dataclass
class StepsConfig:
    annotation: bool = True
    interpretation: bool = True
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
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    interpretation: InterpretationConfig = field(default_factory=InterpretationConfig)
    steps: StepsConfig = field(default_factory=StepsConfig)

    @property
    def out_dir(self) -> Path:
        return Path(self.output_dir) / self.name

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


NEGATIVE_STRATEGIES = ("gc-matched", "shuffle")
NEGATIVE_SOURCES = ("genes_bed", "gtf")
REGION_CLASS_NAMES = ("UTR", "exonic", "intronic")
LEAKAGE_MODES = ("report", "remove")
ASSEMBLIES = ("GRCh38", "GRCh37", "any")
ATTRIBUTION_METHODS = ("ism", "saliency")
BACKGROUNDS = ("sampled", "annotated")
IUPAC = set("ACGUTRYSWKMBDHVN")
INPUT_PATH_FIELDS = ("input", "reference.genome", "reference.genes_bed", "reference.gtf", "negatives.bed",
                     "interpretation.motif_db", "interpretation.background_genes")


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
    n = cfg.negatives
    if n.bed:
        need_file("negatives.bed", n.bed)
    elif n.source == "gtf":
        need_file("reference.gtf (needed for negatives.source: gtf)", cfg.reference.gtf)
    else:
        need_file("reference.genes_bed (needed to generate negatives)", cfg.reference.genes_bed)
    if n.source not in NEGATIVE_SOURCES:
        problems.append(f"negatives.source must be one of {NEGATIVE_SOURCES}, got '{n.source}'")
    if (not isinstance(n.region_classes, list) or not n.region_classes
            or any(c not in REGION_CLASS_NAMES for c in n.region_classes)):
        problems.append(f"negatives.region_classes must be a non-empty list drawn from {list(REGION_CLASS_NAMES)}")
    if n.match_region and n.source != "gtf":
        problems.append("negatives.match_region needs negatives.source: gtf")
    for key in ("match_region", "strand_aware", "qc"):
        if not isinstance(getattr(n, key), bool):
            problems.append(f"negatives.{key} must be true or false")

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

    if cfg.reference.assembly not in ASSEMBLIES:
        problems.append(f"reference.assembly must be one of {ASSEMBLIES}, got '{cfg.reference.assembly}'")
    if cfg.split.leakage not in LEAKAGE_MODES:
        problems.append(f"split.leakage must be one of {LEAKAGE_MODES}, got '{cfg.split.leakage}'")
    if not isinstance(cfg.split.leakage_kmer, int) or cfg.split.leakage_kmer < 8:
        problems.append("split.leakage_kmer must be a whole number >= 8")

    e = cfg.evaluation
    if not isinstance(e.n_seeds, int) or e.n_seeds < 1:
        problems.append("evaluation.n_seeds must be a whole number >= 1")
    if not isinstance(e.bootstrap, int) or e.bootstrap < 0:
        problems.append("evaluation.bootstrap must be a whole number >= 0")
    if not isinstance(e.threshold, (int, float)) or not 0 < e.threshold < 1:
        problems.append("evaluation.threshold must be between 0 and 1")

    m = cfg.model
    for key, minimum in (("max_len", 8), ("epochs", 1), ("patience", 1), ("batch_size", 1)):
        v = getattr(m, key)
        if not isinstance(v, int) or v < minimum:
            problems.append(f"model.{key} must be a whole number >= {minimum}")
    if not isinstance(m.learning_rate, (int, float)) or m.learning_rate <= 0:
        problems.append("model.learning_rate must be a positive number")
    if not isinstance(m.dropout, (int, float)) or not 0 <= m.dropout < 1:
        problems.append("model.dropout must be between 0 and 1")
    if not isinstance(m.weight_decay, (int, float)) or m.weight_decay < 0:
        problems.append("model.weight_decay must be >= 0")
    ks = m.lr_kmers
    if not isinstance(ks, list) or not ks or not all(isinstance(k, int) and 1 <= k <= 8 for k in ks):
        problems.append("model.lr_kmers must be a list of whole numbers between 1 and 8, e.g. [2, 3]")
    if not isinstance(cfg.seed, int):
        problems.append("seed must be a whole number")

    it = cfg.interpretation
    if it.method not in ATTRIBUTION_METHODS:
        problems.append(f"interpretation.method must be one of {ATTRIBUTION_METHODS}, got '{it.method}'")
    for key, lo, hi in (("n_sequences", 1, None), ("kmer_size", 3, 10), ("top_kmers", 1, None),
                        ("streme_nmotifs", 1, None)):
        v = getattr(it, key)
        if not isinstance(v, int) or v < lo or (hi is not None and v > hi):
            problems.append(f"interpretation.{key} must be a whole number >= {lo}" + (f" and <= {hi}" if hi else ""))
    if it.motif_db:
        need_file("interpretation.motif_db", it.motif_db)
    if not isinstance(it.known_motifs, dict):
        problems.append("interpretation.known_motifs must be a mapping of name: IUPAC motif")
    else:
        for k, v in it.known_motifs.items():
            if not isinstance(v, str) or not v or set(v.upper()) - IUPAC:
                problems.append(f"interpretation.known_motifs.{k} must be an IUPAC sequence (ACGU/T, RYSWKMBDHVN)")
    if it.background not in BACKGROUNDS:
        problems.append(f"interpretation.background must be one of {BACKGROUNDS}, got '{it.background}'")
    if it.background_genes:
        need_file("interpretation.background_genes", it.background_genes)
    if not isinstance(it.enrichment_sources, list) or not it.enrichment_sources:
        problems.append("interpretation.enrichment_sources must be a non-empty list, e.g. [GO:BP, REAC]")

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
  genes_bed: genes_only.bed      # regions negatives are drawn from; give it a strand column (rbp-pipeline make-regions)
  gtf: null                      # optional GENCODE GTF for exonic/intronic/UTR annotation
  assembly: GRCh38               # GRCh38 | GRCh37 | any — checked against the genome's chr1 length

peaks:
  min_log2fc: null               # keep peaks with log2 fold-enrichment (BED col 7) >= this; ENCODE uses 3
  min_log10p: null               # keep peaks with -log10 p (BED col 8) >= this; ENCODE uses 3
  window: 100                    # resize every peak to this width around its midpoint; null = as given

negatives:
  strategy: gc-matched           # gc-matched | shuffle
  gc_candidates: 10              # candidate regions per peak for gc-matched
  source: genes_bed              # genes_bed (reference.genes_bed) | gtf (stranded UTR/exonic/intronic from reference.gtf)
  region_classes: [UTR, exonic, intronic]   # with source: gtf, which classes negatives may come from
  match_region: false            # with source: gtf, each peak's negatives come from its own region class
  strand_aware: true             # '+' peaks go into '+' regions, '-' into '-' (needs a strand column in genes_bed)
  qc: true                       # warn if negatives differ from positives in region class or strand (needs gtf)
  bed: null                      # pre-computed negatives BED (skips generation)

split:
  test_chroms: [chr21, chr22, chrX]
  val_chroms: [chr19, chr20]
  leakage: report                # report | remove val/test sequences that resemble training ones
  leakage_kmer: 20               # resemblance = identical, reverse complement, or a shared 20-mer

model:
  max_len: 126                   # CNN input length; longer sequences are centre-cropped
  epochs: 30
  patience: 5                    # early-stopping patience (epochs without val-loss improvement)
  batch_size: 64
  learning_rate: 0.001
  dropout: 0.3
  weight_decay: 0.0              # e.g. 0.0001 to reduce overfitting (uses AdamW)
  lr_kmers: [2, 3]               # k-mer sizes for logistic regression; [3, 4, 5] gives more readable k-mers

evaluation:
  n_seeds: 1                     # e.g. 5: retrain with 5 seeds and report mean ± SD
  bootstrap: 1000                # resamples for 95% confidence intervals (0 = off)
  threshold: 0.5                 # cut-off for precision, recall, F1, MCC, confusion matrix

interpretation:                  # what the models learned (held-out test positives only)
  n_sequences: 500               # top-scoring test positives explained by the CNN
  method: ism                    # ism (in-silico mutagenesis) | saliency (gradient x input, faster)
  kmer_size: 5                   # CNN importance summed over k-mers of this size
  top_kmers: 20
  motifs: true                   # de novo motifs with STREME and Tomtom (MEME Suite); skipped if not installed
  streme_nmotifs: 8
  motif_db: null                 # MEME-format RNA motif database for Tomtom, e.g. Ray2013_rbp_Homo_sapiens.meme
  known_motifs:                  # checked for enrichment in positives vs negatives (name: IUPAC); add your protein's
    GGAC: GGAC
    CAUH: CAUH
    m6A_DRACH: DRACH
    GGUG: GGUG
    UG_repeat: UGUGUG
    ARE_AUUUA: AUUUA
  enrichment: false              # g:Profiler GO/KEGG/Reactome enrichment (needs internet and reference.gtf)
  organism: hsapiens
  enrichment_sources: [GO:BP, GO:MF, GO:CC, KEGG, REAC]
  background: sampled            # sampled: genes overlapping peaks or negatives | annotated: all genes
  background_genes: null         # better: a list of genes expressed in the cell line (one symbol per line)
  protein_coding_only: true      # restrict bound genes and background to protein-coding genes (needs gtf)

steps:
  annotation: true               # needs reference.gtf
  interpretation: true
  report: true                   # Word report (needs Node.js)
"""
