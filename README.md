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
9. **Interprets the models**: in-silico mutagenesis (or saliency) on held-out test peaks, aggregated by position and k-mer; CNN filter motifs; top logistic-regression k-mers; known-motif enrichment; de novo motifs with **STREME** and comparison with known RBP motifs by **Tomtom**; optional **g:Profiler** enrichment of bound genes against an explicit background
10. Writes a predictions table, figures, metrics, a run log, a Word report and full run metadata

## Installation

```bash
git clone https://github.com/akhilesh1904-bioinfo/rbp-analysis-pipeline.git
cd rbp-analysis-pipeline
conda env create -f environment.yml      # Python, PyTorch, bedtools, samtools, MEME Suite, Node.js
conda activate rbp-pipeline
rbp-pipeline --version
```

Without conda: install `bedtools` and `samtools` with your package manager, then `pip install -e ".[dev]"`. MEME Suite (for STREME/Tomtom) is optional; without it those two steps are skipped with a note.

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
| `interpretation/` | CNN position and k-mer importance, filter motifs (`.meme`), LR k-mer coefficients, known-motif enrichment, STREME motifs, Tomtom matches, g:Profiler results, `interpretation_summary.json` |
| `figures/interp_*.png` | importance profile around the peak centre, per-base importance logos for top test peaks, CNN filter logos, top CNN and LR k-mers, STREME motif logos |
| `<name>_report.docx` | summary report with interpretation and methods |
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

## Interpreting the models

The interpretation step (`steps.interpretation`, on by default) asks what sequence features the models use and whether they match what is known about the protein.

| Analysis | Data | Output |
|---|---|---|
| CNN in-silico mutagenesis (`method: ism`) or gradient x input (`saliency`) | top `n_sequences` held-out **test** positives by CNN score | `cnn_position_importance.csv`, `cnn_kmer_importance.csv`, `cnn_importance.npz` |
| CNN first-layer filters as motifs, ranked by influence (drop in mean logit when zeroed) | same | `cnn_filter_motifs.csv` / `.meme` |
| LR coefficients | training set | `lr_kmer_coefficients.csv` |
| Known motifs (IUPAC, Fisher test, BH) | all positives vs matched negatives | `known_motif_enrichment.csv` |
| STREME de novo motifs (RNA alphabet) | all positives vs matched negatives | `streme_motifs.csv`, `streme/` |
| Tomtom against a motif database | STREME motifs + CNN filters | `tomtom_best_matches.csv` |
| g:Profiler (GO, KEGG, Reactome) | genes with peaks vs background | `enrichment_gprofiler.csv` |

Notes:

- **Use `model.lr_kmers: [3, 4, 5]`** if you want readable LR k-mers; the default `[2, 3]` keeps earlier results reproducible but cannot show 4-mers such as GGAC. LR coefficients are conditional on overlapping k-mers, so read them as candidates.
- **Motif database for Tomtom:** download the MEME Suite motif databases (`motif_databases.12.*.tgz` from meme-suite.org) and set `interpretation.motif_db` to `motif_databases/RNA/Ray2013_rbp_Homo_sapiens.meme` (RNAcompete motifs for human RBPs; other files in `RNA/` work too).
- **Known motifs** checked by default (edit `interpretation.known_motifs`): IGF2BP GGAC (Biswas et al. 2019, *Nat Commun*) and CAUH (Hafner et al. 2010, *Cell*), m6A DRACH (IGF2BPs read m6A; Huang et al. 2018, *Nat Cell Biol*), FUS GGUG (Lerga et al. 2001, *J Biol Chem*), TDP-43 UG repeats (Buratti & Baralle 2001, *J Biol Chem*), the AUUUA ARE pentamer.
- **Enrichment background:** genes with eCLIP peaks are long and well expressed, so testing them against every annotated gene mostly recovers "expressed in this cell line". The default background is genes overlapped by any peak or negative region (the universe the peaks were sampled from). Best is a list of genes expressed in the eCLIP cell line (`interpretation.background_genes`, one symbol per line). g:Profiler needs internet access at run time; if the request fails the run continues and records the failure.
- Attributions explain the model, not the biology directly: with GC-matched, region-matched negatives the features the CNN uses are candidates for sequence preference; with shuffled negatives they will largely reflect composition.

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
  attribution.py         CNN ISM / saliency, k-mer and position aggregation, filter motifs, LR k-mers
  motifs.py              known-motif enrichment, MEME format, STREME and Tomtom wrappers
  enrichment.py          g:Profiler gene-set enrichment with an explicit background
  logos.py               sequence logos (matplotlib)
  interpretation.py      runs the interpretation step and writes its tables and figures
  reporting.py           figures, tables, Word report, run metadata
  pipeline.py            runs the steps in order
  cli.py                 `rbp-pipeline` command
tests/                   pytest suite, including end-to-end runs on synthetic genomes
```

## Tests

```bash
pytest
```

Tests run automatically on every push via GitHub Actions. End-to-end tests build small synthetic genomes with planted motifs and need `bedtools` and `samtools`; they check that the CNN's top k-mers and filters recover the planted motif. STREME, Tomtom and g:Profiler are tested with recorded outputs, so the suite needs neither MEME Suite nor internet access.

## Original command-line interface

The original flag-based script still works and maps onto the same pipeline:

```bash
python run_rbp_analysis.py --input peaks.bed --name MY_PROTEIN \
    --genome GRCh38.fa --genes-bed exons.bed --window 100 --skip-report
```

## License

MIT
