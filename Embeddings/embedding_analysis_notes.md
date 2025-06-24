# Embedding Analysis Script

This script analyzes ICD11 embeddings from different models and creates visualizations grouped by ICD11 chapters, with comprehensive quality evaluation metrics.

## Features

The script contains 6 main functions:

1. **`load_and_process_embeddings()`** - Loads CSV files, converts embeddings to numpy arrays, and extracts chapter numbers
2. **`create_tsne_visualization()`** - Creates t-SNE visualizations grouped by chapter
3. **`create_umap_visualization()`** - Creates UMAP visualizations grouped by chapter  
4. **`analyze_embeddings()`** - Meta function that controls functions 1-6
5. **`evaluate_embedding_quality()`** - Evaluates embedding quality using multiple clustering metrics
6. **`evaluate_dimensionality_reduction_quality()`** - Evaluates t-SNE and UMAP quality using trustworthiness and continuity

## Quality Metrics Included

### Embedding Quality Metrics
- **Silhouette Score**: Measures cluster separation and cohesion (-1 to 1, higher is better)
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster dispersion (higher is better)
- **Davies-Bouldin Index**: Average similarity ratio of clusters (lower is better)
- **Cluster Separation Ratio**: Inter-cluster to intra-cluster distance ratio (higher is better)
- **Chapter Entropy**: Information-theoretic measure of chapter distribution
- **Intra/Inter-cluster Distances**: Average distances within and between chapters

### Dimensionality Reduction Quality Metrics
- **Trustworthiness**: How well close points in low-dim are actually close in high-dim (0 to 1, higher is better)
- **Continuity**: How well close points in high-dim remain close in low-dim (0 to 1, higher is better)
- **Quality Score**: Combined trustworthiness and continuity measure
- **Reduced Space Silhouette**: Silhouette score in the 2D visualization space

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from embedding_analysis import analyze_embeddings

# Run complete analysis (loads data, evaluates quality, creates visualizations)
analyze_embeddings()
```

### Advanced Usage

```python
from embedding_analysis import *

# Load embeddings data
embeddings_data = load_and_process_embeddings("Resulting-embeddings")

# Evaluate embedding quality only
quality_metrics, quality_df, rankings_df = evaluate_embedding_quality(embeddings_data)

# Evaluate dimensionality reduction quality only
dr_metrics, dr_df, dr_rankings_df = evaluate_dimensionality_reduction_quality(embeddings_data)

# Create only specific visualizations
create_tsne_visualization(embeddings_data, save_plots=True)
create_umap_visualization(embeddings_data, save_plots=False)
```

## Input Data Format

The script expects CSV files in the following format:
- Column 1: `ICD11_code` - String containing the ICD11 code
- Column 2: `Vector` - String representation of the embedding vector (e.g., "[1.2, -0.5, 3.1, ...]")

## Output Files for Report Appendix

### CSV Files with Metrics and Rankings
- **`embedding_quality_metrics.csv`** - Raw quality metrics for all models
- **`embedding_quality_rankings.csv`** - Rankings and overall scores for quality metrics
- **`dimensionality_reduction_metrics.csv`** - t-SNE and UMAP quality metrics
- **`dimensionality_reduction_rankings.csv`** - Rankings for dimensionality reduction quality

### Visualizations
- **`embedding_quality_analysis.png`** - Comprehensive quality analysis dashboard
- **Individual model PNG files** - t-SNE and UMAP plots for each model
- **Color scheme**: Uses the specified color palette: '#FF6F61', '#D2B48C', '#AE65D6', '#598A4F', etc.

## Report Summary Output

The script provides a comprehensive console summary including:
1. **Embedding Quality Metrics Table** - All calculated metrics for each model
2. **Quality Rankings Table** - Ranked performance across different metrics
3. **Dimensionality Reduction Quality** - t-SNE and UMAP performance metrics
4. **Top Performing Models** - Best models for each metric category
5. **Generated Files List** - All output files for appendix inclusion

## File Structure

```
Embeddings/
├── embedding_analysis.py              # Main analysis script
├── requirements.txt                   # Dependencies
├── README_embedding_analysis.md       # This file
├── embedding_quality_metrics.csv      # Quality metrics (generated)
├── embedding_quality_rankings.csv     # Quality rankings (generated)
├── dimensionality_reduction_metrics.csv # DR metrics (generated)
├── dimensionality_reduction_rankings.csv # DR rankings (generated)
├── embedding_quality_analysis.png     # Quality dashboard (generated)
└── Resulting-embeddings/              # Folder containing CSV files
    ├── gatortron_ICD11_embeddings.csv
    ├── fasttext_ICD11_embeddings.csv
    ├── tfidf_ICD11_embeddings.csv
    ├── pubmedbert_ICD11_embeddings.csv
    ├── bioclinicalbert_ICD11_embeddings.csv
    ├── biobert_ICD11_embeddings.csv
    └── bert_ICD11_embeddings.csv
```

## Interpretation Guide

### Quality Metrics Interpretation
- **Higher is Better**: Silhouette Score, Calinski-Harabasz Index, Cluster Separation Ratio, Trustworthiness, Continuity
- **Lower is Better**: Davies-Bouldin Index, Mean Intra-cluster Distance, Chapter Entropy
- **Overall Ranking**: Composite score considering all metrics (1 = best performing model)

### Statistical Significance
The script provides rankings that can be used for statistical comparison and model selection based on multiple quality criteria.

## Chapter Mapping

The script uses the provided `get_chapter_from_code()` function to map ICD11 codes to their respective chapters (0-26). 