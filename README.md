# RBP Analysis Pipeline

A generalised pipeline for training and evaluating sequence-based binding models from eCLIP data. Given an ENCODE-style eCLIP BED file (GRCh38), the pipeline trains CNN and logistic regression models on that protein's own binding sites and evaluates how well sequence alone predicts binding.

## What it does

1. Validates your eCLIP BED file (accepts `chr1` or `1` naming; optional ENCODE significance filters)
2. Generates negatives matched to each peak's length, chromosome and GC content (bedtools shuffle within genes, then closest-GC selection)
3. Extracts sequences in the RNA sense (minus-strand peaks are reverse-complemented)
4. Trains CNN and logistic regression models on a chromosome-based split
5. Evaluates on held-out chromosomes and reports AUC alongside a GC-content-only baseline
6. Annotates peaks as exonic/intronic/UTR/intergenic (if a GTF is given)
7. Outputs predictions CSV, ROC curve, score plots, Word report and `run_info.json`

## Requirements

### Python
```bash
pip install -r requirements.txt
```

### System tools
```bash
sudo apt install samtools bedtools nodejs npm
```
The Node.js `docx` package for the Word report is installed automatically on first run.

### Reference files
- GRCh38 genome FASTA with `chr`-prefixed names (e.g. GENCODE `GRCh38.primary_assembly.genome.fa`)
- Gene intervals BED (`genes_only.bed`) for constraining negatives, e.g. from the GENCODE GTF:
  ```bash
  awk -F'\t' '$3=="gene"{print $1"\t"$4-1"\t"$5"\t.\t0\t"$7}' gencode.gtf > genes_only.bed
  ```
- *(Optional)* GENCODE GTF for region annotation

## Usage

```bash
python run_rbp_analysis.py \
    --input      MY_PROTEIN_peaks.bed \
    --name       MY_PROTEIN \
    --genome     /path/to/GRCh38.fa \
    --genes-bed  /path/to/genes_only.bed \
    --gtf        /path/to/gencode.gtf \
    --output-dir ./results
```

Useful options:

| Option | Default | Purpose |
|---|---|---|
| `--test-chroms` | `chr21 chr22 chrX` | Held-out test chromosomes |
| `--val-chroms` | `chr19 chr20` | Validation chromosomes (early stopping) |
| `--min-log2fc`, `--min-log10p` | off | Keep peaks with log2FC (col 7) / −log10 p (col 8) above threshold; ENCODE uses 3 and 3 |
| `--negative-strategy` | `gc-matched` | `gc-matched`: draw several candidates per peak and keep the closest in GC. `shuffle`: one random length/chromosome-matched region, GC uncontrolled |
| `--gc-candidates` | `10` | Candidates per peak for `gc-matched` |
| `--window` | off | Resize every peak to a fixed width around its midpoint (e.g. `100`) |
| `--max-len` | `126` | CNN input length; longer sequences are centre-cropped |
| `--negatives` | — | Use a pre-computed negatives BED (then `--genes-bed` is not needed) |
| `--seed` | `42` | Seed for negatives, splits and model training |
| `--skip-annotation`, `--skip-report` | — | Skip GTF annotation / Word report |

## Why GC matching and the GC baseline matter

eCLIP peaks often sit in exons and UTRs, which are GC-richer than the introns that dominate randomly sampled genic background. With unmatched negatives, a model can score a high AUC simply by detecting GC content. Every run therefore prints the mean GC of positives and negatives and a **GC-only baseline AUC** (each test sequence scored by its GC fraction alone). A model is only learning something beyond composition to the extent that it beats this baseline.

## Interpreting the predictions CSV

Every peak is scored, and the `split` column says where it came from. **Only `split == test` peaks were unseen during training** — scores for `train`/`val` peaks (`seen_in_training = True`) are optimistic and should not be used as evidence of performance.

## Tests

```bash
pip install pytest
pytest
```
The end-to-end test builds a small synthetic genome and runs the full pipeline; it needs `bedtools` and `samtools`.

## License
MIT
