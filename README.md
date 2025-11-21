# EPI Benchmarking Suite

A comprehensive benchmarking framework for evaluating and comparing the performance of enhancer-promoter interaction (EPI) prediction tools across multiple human cell lines.

## Overview

This repository provides a standardized evaluation pipeline for comparing state-of-the-art EPI prediction methods. We benchmark three computational approaches:

- **[eModule](https://github.com/LinjieCodes/eModule)**: A comprehensive framework designed to analyze enhancer expression and identify enhancer-mediated gene regulatory modules using transcriptomic data
- **[TargetFinder](https://github.com/shwhalen/targetfinder)**: An ensemble learning method that integrates hundreds of genomics datasets
- **[EPIPDLF](https://github.com/xzc196/EPIPDLF)**: A pre-trained deep learning framework for EPI prediction


## Installation

### Prerequisites

- Python 3.7+
- Required packages listed in `requirements.txt`

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/epi-benchmarking.git
cd epi-benchmarking

# Install dependencies
pip install -r requirements.txt

# Download required data files
# (Data download instructions to be added based on your data hosting)
```

## Repository Structure

```
epi-benchmarking/
├── evaluate_eModule.py          # eModule evaluation script
├── evaluate_targetFinder.py     # TargetFinder evaluation script  
├── evaluate_EPIPDLF.py           # EPIPDLF model evaluation
├── plot_performance.py         # Performance visualization
├── data_loader.py              # Data loading utilities
├── config.py                   # Configuration parameters
├── model_dnabert2_embedding_small.py  # EPIPDLF model architecture
├── optimized_data_preprocessing.py    # Data preprocessing utilities
└── README.md                   # This file
```

## Usage

### 1. Running Evaluations

#### Evaluate eModule
```bash
python evaluate_eModule.py
```

This script:
- Loads eModule prediction results for all six cell lines
- Computes F1-scores, precision, and recall metrics
- Generates plots

#### Evaluate TargetFinder
```bash
python evaluate_targetFinder.py
```

This script:
- Performs 10-fold cross-validation for each cell line
- Uses GradientBoostingClassifier with optimized parameters
- Computes mean performance metrics across folds
- Generates ROC and PR curve visualizations

#### Evaluate EPIPDLF
```bash
python evaluate_EPIPDLF.py
```

This script:
- Loads the pre-trained EPIPDLF model
- Evaluates performance on test datasets
- Computes comprehensive performance metrics

### 2. Generate Performance Visualizations

```bash
python plot_performance.py
```

Generates heatmaps showing:
- F1-score comparison across methods and cell lines
- Precision comparison
- Recall comparison

## Performance Metrics

### Evaluation Results

| Method | Average F1-Score | Average Precision | Average Recall |
|--------|------------------|-------------------|----------------|
| eModule | 0.840 | 0.860 | 0.825 |
| TargetFinder | 0.691 | 0.898 | 0.565 |
| EPIPDLF | 0.828 | 0.848 | 0.738 |

*Results averaged across six cell lines (GM12878, IMR90, HeLa-S3, HUVEC, K562, NHEK)*

### Key Findings

1. **EPIPDLF** and **eModule** show competitive performance with good balance between precision and recall
2. **TargetFinder** achieves the highest precision but suffers from low recall
3. Performance varies significantly across cell lines, highlighting the importance of cell-type specific evaluation

## Method Descriptions

### eModule
Comprehensive framework designed to analyze enhancer expression and identify enhancer-mediated gene regulatory modules using transcriptomic data.

### TargetFinder  
Ensemble learning method that integrates hundreds of genomics datasets to identify predictive features. Key insight: marks on looping chromatin (window between enhancer and promoter) are more predictive than proximal marks.

### EPIPDLF
Pre-trained deep learning framework using DNA sequence information. Incorporates:
- DNA sequence encoding with k-mer features
- Convolutional neural networks for feature extraction
- Multi-head attention mechanisms
- Bidirectional GRU layers for sequence modeling

## Data Sources

The benchmarking uses publicly available datasets from:
- ENCODE Project
- 4D Nucleome Project  
- Hi-C interaction data from Rao et al. (2014)
- ChIP-seq data for various histone modifications and transcription factors


## License

This project is licensed under the Apache 2.0 License - see LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact [JL.linjie@outlook.com]


---

## Technical Details

### Model Architectures

#### EPIPDLF Architecture
```python
# Key components:
- DNA sequence embedding (DNA-BERT based)
- Convolutional layers for feature extraction
- Multi-head attention mechanism
- Bidirectional GRU for sequence modeling
- Binary classification head
```

#### TargetFinder Features
- DNA methylation data
- Histone modification marks (H3K27ac, H3K4me1, H3K4me3)
- Transcription factor binding sites
- Chromatin accessibility (DNase-seq)
- CTCF and cohesin binding

### Evaluation Protocol

All methods are evaluated using:
- **Stratified sampling** to maintain class balance
- **Fixed random seeds** for reproducible results
- **Multiple metrics** (F1, precision, recall) for comprehensive assessment

### System Requirements

- **Memory**: 16GB RAM minimum, 32GB recommended
- **Storage**: 10GB for datasets and model files
- **GPU**: Optional but recommended for EPIPDLF evaluation (CUDA-compatible)

---

*Last updated: November 2025*