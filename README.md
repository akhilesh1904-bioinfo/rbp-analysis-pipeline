# RBP Analysis Pipeline

A generalised pipeline for training and evaluating sequence-based binding models from eCLIP data. Given any eCLIP BED file (GRCh38), the pipeline trains CNN and logistic regression models on that protein's own binding sites and evaluates how well sequence alone predicts binding.

## What it does

1. Validates your eCLIP BED file
2. Generates matched negatives via bedtools shuffle
3. Extracts sequences from GRCh38
4. Trains CNN and logistic regression models on a chromosome-based split
5. Evaluates on held-out chromosomes and reports AUC
6. Annotates peaks as exonic/intronic/UTR/intergenic
7. Outputs predictions CSV, ROC curve, score plots, and Word report

## Requirements

### Python
pip install -r requirements.txt

### System tools
sudo apt install samtools bedtools nodejs npm
npm install --prefix . docx

### Reference files
- GRCh38.primary_assembly.genome.fa (ENCODE)
- gencode.v49.primary_assembly.annotation.gtf (GENCODE)
- genes_only.bed (generate from GTF — see docs)

## Usage

python run_rbp_analysis.py \
    --input     MY_PROTEIN_peaks.bed \
    --name      MY_PROTEIN \
    --genome    /path/to/GRCh38.fa \
    --gtf       /path/to/gencode.gtf \
    --genes-bed /path/to/genes_only.bed

## License
MIT
