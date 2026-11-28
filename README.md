# RBP Analysis Pipeline

[![tests](https://github.com/akhilesh1904-bioinfo/rbp-analysis-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/akhilesh1904-bioinfo/rbp-analysis-pipeline/actions/workflows/tests.yml)

Train and evaluate sequence-based models of RNA-binding protein (RBP) binding from ENCODE eCLIP peaks, with negatives that control for length, chromosome and GC content, so reported performance reflects sequence preference rather than composition bias.

For each protein the pipeline:

1. Validates the eCLIP BED (accepts `chr1` or `1` naming; optional ENCODE significance filters; optional fixed-width window)
2. Generates negatives matched to each peak's **length, chromosome, strand and GC content** (and optionally its **region class**: UTR, exonic or intronic), each lying entirely inside a region set you choose, then **checks** that negatives resemble positives in region and strand
3. Extracts sequences in the **RNA sense** (minus-strand peaks are reverse-complemented)
4. Trains a 1D CNN and a k-mer logistic regression on a **chromosome-held-out split**
5. Evaluates on the test set with ROC-AUC, PR-AUC, precision, recall, F1, MCC, Brier score, confusion matrices and calibration, with **bootstrap 95% CIs** that resample each peak together with its matched negative (or all pairs from one gene), **paired significance tests** (CNN vs LR vs a **GC-content-only baseline**) and optional **multi-seed** mean ± SD
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
rbp-pipeline default-config > my_protein.yaml     # commented template
# edit name, input and reference paths
rbp-pipeline validate my_protein.yaml             # checks every setting and file before running
rbp-pipeline make-regions --gtf genes.gtf --out-dir ~/ref   # optional: stranded UTR/exonic/intronic BEDs
rbp-pipeline run my_protein.yaml
```

Any setting can be overridden for a single run:

```bash
rbp-pipeline run my_protein.yaml --set negatives.strategy=shuffle --set name=MY_PROTEIN_shuffle
```

See [`example/config.yaml`](example/config.yaml) for a real configuration and [`config/default.yaml`](config/default.yaml) for every option.

### Reference files

| File | Use | How to get it |
|---|---|---|
| Genome FASTA | sequences | GENCODE `GRCh38.primary_assembly.genome.fa` (chr-prefixed names) |
| Region BED for negatives | where negatives are drawn from | see below |
| GTF (optional) | region annotation | GENCODE `gencode.v*.primary_assembly.annotation.gtf` |

Choose the negative region set to match where your protein binds: proteins that bind mature mRNA need UTR/exon negatives, intron-binding proteins need intron negatives. The simplest option is to let the pipeline build stranded region sets from the GTF (`negatives.source: gtf`, see [Strand and region matching](#strand-and-region-matching)). To make reusable stranded BEDs for `reference.genes_bed` instead:

```bash
rbp-pipeline make-regions --gtf gencode.v49.primary_assembly.annotation.gtf --out-dir ~/ref
# writes *.UTR.stranded.bed, *.exonic.stranded.bed, *.intronic.stranded.bed and *.genic.stranded.bed
```

Region BEDs need a strand in column 6; unstranded files (for example exons merged across strands) let negatives land antisense to genes and trigger a warning.

## Outputs

Results go to `<output_dir>/<name>/`:

| File | Contents |
|---|---|
| `<name>_predictions.csv` | every peak with CNN/LR scores, `split`, `seen_in_training`, region and gene |
| `<name>_metrics.json` / `.csv` | all test metrics with 95% CIs, paired model comparisons, seed mean ± SD, leakage summary |
| `leakage_flags.csv` | val/test sequences resembling training sequences, with the reason |
| `negatives_qc.csv` | region-class and antisense fractions of positives vs negatives |
| `unmatched_peaks.csv` | peaks that got no matched negative (and so are left out of the paired splits), with their GC |
| `figures/` | ROC and precision-recall curves (with GC baseline), calibration, confusion matrices, score distributions, CNN training curve |
| `interpretation/` | CNN position and k-mer importance, filter motifs (`.meme`), LR k-mer coefficients, known-motif enrichment, STREME motifs, Tomtom matches, g:Profiler results, `interpretation_summary.json` |
| `figures/interp_*.png` | importance profile around the peak centre, per-base importance logos for top test peaks, CNN filter logos, top CNN and LR k-mers, STREME motif logos |
| `<name>_report.docx` | summary report with interpretation and methods |
| `pipeline.log`, `run_info.json` | full log; exact config and software versions |
| `positives.fa`, `negatives.fa`, `*_negatives.bed`, `<name>_cnn.pt` | intermediate data and model weights |

Only peaks with `split == test` were unseen during training. Scores for train/val peaks are optimistic and should not be used as evidence of performance.

## Why GC matching and the GC baseline matter

eCLIP peaks often lie in exons and UTRs, which are GC-richer than the introns that dominate randomly sampled genic background. With unmatched negatives, a model can score a high AUC simply by detecting GC content. Every run prints the mean GC of positives and negatives and a GC-only baseline AUC. A model is learning something beyond composition only to the extent that it beats this baseline.

In practice the difference is large: with negatives shuffled anywhere within genes, GC content alone can reach an AUC around 0.8 for exon- and UTR-binding proteins, so a model's headline AUC says little until it is compared with this baseline. With GC-, strand- and region-matched negatives the baseline falls to about 0.5, and whatever the models achieve above it is sequence signal.

## Strand and region matching

GC is not the only shortcut. Two more are checked and controlled:

- **Strand.** Sequences are read on the peak's strand. If a negative region is placed on a gene on the other strand, its sequence is the reverse complement of a transcript, which a model can tell apart from real mRNA. With `negatives.strand_aware: true` (default) '+' peaks are only placed in '+' regions and '-' peaks in '-' regions. This needs a strand in column 6 of `reference.genes_bed`; an unstranded file (for example exons merged across strands) triggers a warning. Build stranded files once with:

  ```bash
  rbp-pipeline make-regions --gtf ~/ref/gencode.v49.primary_assembly.annotation.gtf --out-dir ~/ref
  ```

  or let the pipeline build them with `negatives.source: gtf`.
- **Region class.** Peaks of mRNA-binding proteins are mostly UTR; exon-drawn negatives are mostly coding sequence. With `negatives.source: gtf` and `negatives.match_region: true`, each peak's negatives come from its own class (UTR, exonic = exon minus UTR, intronic = gene minus exons), on the same strand. Intergenic peaks draw from all chosen classes.
- Every placement lies **entirely** inside a region (bedtools `shuffle -incl` alone only guarantees the start).
- **QC.** With `reference.gtf` set, negatives are annotated like positives and the run prints both region breakdowns, the share of negatives antisense to a gene, the share in their peak's own class, and a warning when they differ (`negatives_qc.csv`, `negatives_qc` in the metrics JSON).

```yaml
negatives:
  strategy: gc-matched
  gc_candidates: 30
  source: gtf
  region_classes: [UTR, exonic]     # mRNA-binding protein; add intronic for intron binders
  match_region: true
```

## Recommended settings for reporting results

```yaml
evaluation:
  n_seeds: 5          # mean ± SD over 5 CNN trainings
  bootstrap: 1000     # 95% CIs and paired tests
  bootstrap_unit: gene   # resample whole genes (needs reference.gtf)
split:
  leakage: remove     # compare with `report` to see whether near-duplicates inflate performance
model:
  weight_decay: 0.0001   # optional: reduces CNN overfitting (check the training curve)
```

A model comparison counts as significant only when the paired bootstrap interval for the AUC difference excludes zero. With a few hundred test sequences, AUC differences below about 0.03 are usually within noise.

### Matched pairs and uncertainty

Each negative is chosen for one peak, so a peak and its negative are not independent observations, and several peaks often fall in the same gene or UTR. Treating every test sequence as independent therefore makes confidence intervals too narrow.

- `split.balance: paired` (default) keeps each peak together with its own negative in every split. Peaks without a negative (no usable candidate, or its negative removed by the leakage check) and negatives whose peak is gone are dropped, so the test set is made of complete pairs. `random` downsamples the larger class at random, as before v0.6.0.
- `evaluation.bootstrap_unit` sets what the bootstrap resamples: `pair` (default; a peak with its negative), `gene` (all pairs whose peak lies in one gene; peaks outside genes count as their own unit; needs `reference.gtf`) or `sequence` (each sequence on its own, as before v0.6.0). The paired model comparisons use the same unit. `gene` gives the most conservative intervals.
- Every run reports how many peaks got no matched negative, their GC compared with matched peaks, and the GC difference within pairs (`matching` in the metrics JSON, `unmatched_peaks.csv`). If many peaks are unmatched, and especially if they differ in GC, the test set no longer represents all peaks: add candidates (`negatives.gc_candidates`) or widen the region classes.
- `negatives.max_gc_diff` (e.g. `0.05`) refuses candidates more than that GC fraction from their peak. Matches become tighter, at the price of more unmatched peaks; compare the two counts before choosing.

Intervals still describe resampling of one test set on fixed test chromosomes; they do not include the variation from choosing different test chromosomes.

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
- **Known motifs** checked by default: a small panel of RNA sequence elements frequently reported for RBPs — GGAC (Biswas et al. 2019, *Nat Commun*), CAUH (Hafner et al. 2010, *Cell*), the m6A consensus DRACH (Huang et al. 2018, *Nat Cell Biol*), GGUG (Lerga et al. 2001, *J Biol Chem*), UG repeats (Buratti & Baralle 2001, *J Biol Chem*) and the AUUUA ARE pentamer. Replace or extend it per protein with `interpretation.known_motifs` (name: IUPAC sequence).
- **Enrichment background:** genes with eCLIP peaks are long and well expressed, so testing them against every annotated gene mostly recovers "expressed in this cell line". The default background is genes overlapped by any peak or negative region (the universe the peaks were sampled from). Best is a list of genes expressed in the eCLIP cell line (`interpretation.background_genes`, one symbol per line). Query and background are restricted to protein-coding genes (`protein_coding_only`), the background is applied over annotated genes only (g:Profiler `custom_annotated`), and the run logs the effective background size g:Profiler used, warning if it is much larger than the list sent. g:Profiler needs internet access at run time; if the request fails the run continues and records the failure.
- Attributions explain the model, not the biology directly: with GC-matched, region-matched negatives the features the CNN uses are candidates for sequence preference; with shuffled negatives they will largely reflect composition.

## Project structure

```
src/rbp_pipeline/
  config.py              YAML config, overrides, validation
  genome.py              chromosome naming, FASTA access, strand-aware fetching
  preprocessing.py       BED parsing, filters, windowing
  negative_sampling.py   length/chromosome/strand(/region)-matched candidates, GC matching, region QC
  sequence_extraction.py RNA-sense sequences, FASTA output
  dataset.py             chromosome splits, one-hot and k-mer encodings
  models.py              CNN and logistic regression
  training.py            training + test-set scoring
  evaluation.py          metrics, bootstrap CIs, paired comparisons, seed summaries
  leakage.py             exact / reverse-complement / shared k-mer overlap between splits
  annotation.py          strand-aware UTR/exonic/intronic annotation, stranded region sets (bedtools)
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
