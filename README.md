# RBP Analysis Pipeline

[![tests](https://github.com/akhilesh1904-bioinfo/rbp-analysis-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/akhilesh1904-bioinfo/rbp-analysis-pipeline/actions/workflows/tests.yml)

Train and evaluate sequence-based models of RNA-binding protein (RBP) binding from ENCODE eCLIP peaks, with negatives that control for length, chromosome and GC content, so reported performance reflects sequence preference rather than composition bias.

For each protein the pipeline:

1. Validates the eCLIP BED (accepts `chr1` or `1` naming; optional ENCODE significance filters; optional fixed-width window)
2. Generates negatives matched to each peak's **length, chromosome and GC content**, drawn from a region set you choose (genes, exons or introns)
3. Extracts sequences in the **RNA sense** (minus-strand peaks are reverse-complemented)
4. Trains a 1D CNN and a k-mer logistic regression on a **chromosome-held-out split**
5. Evaluates on the test set with ROC-AUC, PR-AUC, precision, recall, F1, MCC, Brier score, confusion matrices and calibration, with **bootstrap 95% CIs**, **paired significance tests** (CNN vs LR vs a **GC-content-only baseline**) and optional **multi-seed** mean ± SD
6. Checks for **sequence leakage**: val/test sequences identical to, reverse-complementary to, or sharing a 20-mer with training sequences (report or remove)
7. Validates the **genome build** (GRCh38 vs GRCh37 by chr1 length) and flags peaks that run past chromosome ends
8. Annotates peaks as UTR / exonic / intronic / intergenic, **strand-aware**
9. Writes a predictions table, figures, metrics, a run log, a Word report and full run metadata

## Installation

```bash
git clone https://github.com/akhilesh1904-bioinfo/rbp-analysis-pipeline.git
cd rbp-analysis-pipeline
conda env create -f environment.yml      # Python, PyTorch, bedtools, samtools, Node.js
conda activate rbp-pipeline
rbp-pipeline --version
```

Without conda: install `bedtools` and `samtools` with your package manager, then `pip install -e ".[dev]"`.

## Quick start

```bash
rbp-pipeline default-config > igf2bp2.yaml     # commented template
# edit name, input and reference paths
rbp-pipeline validate igf2bp2.yaml             # checks every setting and file before running
rbp-pipeline run igf2bp2.yaml
```

Any setting can be overridden for a single run:

```bash
rbp-pipeline run igf2bp2.yaml --set negatives.strategy=shuffle --set name=IGF2BP2_shuffle
```

See [`example/config.yaml`](example/config.yaml) for a real configuration and [`config/default.yaml`](config/default.yaml) for every option.

### Reference files

| File | Use | How to get it |
|---|---|---|
| Genome FASTA | sequences | GENCODE `GRCh38.primary_assembly.genome.fa` (chr-prefixed names) |
| Region BED for negatives | where negatives are drawn from | see below |
| GTF (optional) | region annotation | GENCODE `gencode.v*.primary_assembly.annotation.gtf` |

Choose the negative region set to match where your protein binds. Exon-binding proteins (e.g. IGF2BP2) need exon-only negatives; intron-binding proteins (e.g. FUS) need intron-only negatives:

```bash
GTF=gencode.v49.primary_assembly.annotation.gtf
MAIN='$1 ~ /^chr([0-9]+|X|Y)$/'
awk -F'\t' "$MAIN && \$3==\"gene\" {print \$1\"\t\"\$4-1\"\t\"\$5}" $GTF | sort -k1,1 -k2,2n | bedtools merge > genes.bed
awk -F'\t' "$MAIN && \$3==\"exon\" {print \$1\"\t\"\$4-1\"\t\"\$5}" $GTF | sort -k1,1 -k2,2n | bedtools merge > exons.bed
bedtools subtract -a genes.bed -b exons.bed > introns.bed
```

## Outputs

Results go to `<output_dir>/<name>/`:

| File | Contents |
|---|---|
| `<name>_predictions.csv` | every peak with CNN/LR scores, `split`, `seen_in_training`, region and gene |
| `<name>_metrics.json` / `.csv` | all test metrics with 95% CIs, paired model comparisons, seed mean ± SD, leakage summary |
| `leakage_flags.csv` | val/test sequences resembling training sequences, with the reason |
| `figures/` | ROC and precision-recall curves (with GC baseline), calibration, confusion matrices, score distributions, CNN training curve |
| `<name>_report.docx` | summary report with methods |
| `pipeline.log`, `run_info.json` | full log; exact config and software versions |
| `positives.fa`, `negatives.fa`, `*_negatives.bed`, `<name>_cnn.pt` | intermediate data and model weights |

Only peaks with `split == test` were unseen during training. Scores for train/val peaks are optimistic and should not be used as evidence of performance.

## Why GC matching and the GC baseline matter

eCLIP peaks often lie in exons and UTRs, which are GC-richer than the introns that dominate randomly sampled genic background. With unmatched negatives, a model can score a high AUC simply by detecting GC content. Every run prints the mean GC of positives and negatives and a GC-only baseline AUC. A model is learning something beyond composition only to the extent that it beats this baseline.

Example (IGF2BP2, K562, 3,834 IDR peaks, held-out chr21/chr22/chrX):

| Negatives | GC-only AUC | CNN AUC | LR AUC |
|---|---|---|---|
| Random within genes | 0.815 | 0.893 | 0.850 |
| GC-matched, exons only, 100 bp window | 0.509 | 0.794 | 0.756 |

## Recommended settings for reporting results

```yaml
evaluation:
  n_seeds: 5          # mean ± SD over 5 CNN trainings
  bootstrap: 1000     # 95% CIs and paired tests
split:
  leakage: remove     # compare with `report` to see whether near-duplicates inflate performance
model:
  weight_decay: 0.0001   # optional: reduces CNN overfitting (check the training curve)
```

A model comparison counts as significant only when the paired bootstrap interval for the AUC difference excludes zero. With a few hundred test sequences, AUC differences below about 0.03 are usually within noise.

## Project structure

```
src/rbp_pipeline/
  config.py              YAML config, overrides, validation
  genome.py              chromosome naming, FASTA access, strand-aware fetching
  preprocessing.py       BED parsing, filters, windowing
  negative_sampling.py   length/chromosome-matched candidates, GC matching
  sequence_extraction.py RNA-sense sequences, FASTA output
  dataset.py             chromosome splits, one-hot and k-mer encodings
  models.py              CNN and logistic regression
  training.py            training + test-set scoring
  evaluation.py          metrics, bootstrap CIs, paired comparisons, seed summaries
  leakage.py             exact / reverse-complement / shared k-mer overlap between splits
  annotation.py          strand-aware UTR/exonic/intronic annotation (bedtools)
  reporting.py           figures, tables, Word report, run metadata
  pipeline.py            runs the steps in order
  cli.py                 `rbp-pipeline` command
tests/                   pytest suite, including end-to-end runs on synthetic genomes
```

## Tests

```bash
pytest
```

Tests run automatically on every push via GitHub Actions. End-to-end tests build small synthetic genomes with planted motifs and need `bedtools` and `samtools`.

## Original command-line interface

The original flag-based script still works and maps onto the same pipeline:

```bash
python run_rbp_analysis.py --input peaks.bed --name MY_PROTEIN \
    --genome GRCh38.fa --genes-bed exons.bed --window 100 --skip-report
```

## License

MIT
