# Music Clustering with Variational Autoencoders

Unsupervised learning pipeline for clustering hybrid language music tracks using Variational Autoencoders (VAE).

## Project Overview

This project implements three difficulty levels of music clustering:
- **Easy Task**: Basic VAE with MFCC features
- **Medium Task**: Hybrid ConvVAE combining audio spectrograms and text embeddings
- **Hard Task**: Conditional VAE (CVAE) with audio, text, and genre/language labels

## Project Structure

```
MusicClusteringProject/
├── src/
│   ├── model.py              # VAE model architectures
│   ├── preprocess.py          # MFCC feature extraction (Easy task)
│   ├── preprocess2.py         # Spectrogram extraction (Medium/Hard tasks)
│   ├── hybrid_data.py         # Hybrid data preparation (audio + text)
│   ├── train.py               # Easy task training script
│   ├── train_medium.py        # Medium task training script
│   ├── train_hard.py          # Hard task training script
│   ├── clustering.py          # Clustering algorithms (K-Means, Agglomerative, DBSCAN)
│   ├── evaluation.py          # Evaluation metrics
│   └── plot_metrics_summary.py # Metrics visualization
├── notebooks/
│   └── exploratory.ipynb      # Exploratory data analysis
├── dataset/
│   └── audio/                 # Audio files organized by language
│       ├── bangla/
│       ├── english/
│       ├── hindi/
│       ├── korean/
│       ├── japanese/
│       └── spanish/
├── results/
│   ├── clustering_metrics.csv # All calculated metrics
│   └── latent_visualization/  # Visualization plots
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/404sanchita/Music-Clustering.git
cd Music-Clustering
```

2. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Easy Task

1. Extract MFCC features:
```bash
python src/preprocess.py
```

2. Train VAE and perform clustering:
```bash
python src/train.py
```

### Medium Task

1. Extract spectrograms:
```bash
python src/preprocess2.py
```

2. Create hybrid data (audio + text):
```bash
python src/hybrid_data.py
```

3. Train HybridConvVAE and perform clustering:
```bash
python src/train_medium.py
```

### Hard Task

1. Ensure hybrid data exists (from Medium task step 2)

2. Train Conditional VAE and compare with baselines:
```bash
python src/train_hard.py
```

### Generate Metrics Summary

After running training scripts, generate summary visualizations:
```bash
python src/plot_metrics_summary.py
```

## Models

- **MusicVAE**: Basic VAE for MFCC features (Easy task)
- **ConvVAE**: Convolutional VAE for spectrograms
- **HybridConvVAE**: Combines audio spectrograms with text embeddings (Medium task)
- **HardMusicCVAE**: Conditional VAE with audio, text, and labels (Hard task)
- **SimpleAutoencoder**: Baseline autoencoder for comparison

## Clustering Algorithms

- **K-Means**: Centroid-based clustering
- **Agglomerative Clustering**: Hierarchical clustering
- **DBSCAN**: Density-based clustering

## Evaluation Metrics

- **Silhouette Score**: Measures cluster cohesion and separation
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance
- **Davies-Bouldin Index**: Average similarity between clusters
- **Adjusted Rand Index (ARI)**: Agreement with ground truth labels
- **Normalized Mutual Information (NMI)**: Mutual information between clusters and labels
- **Cluster Purity**: Fraction of dominant class in each cluster

## Results

Results are saved in:
- `results/clustering_metrics.csv`: All calculated metrics
- `results/latent_visualization/`: Visualization plots (t-SNE, clustering comparisons)

## Requirements

See `requirements.txt` for full list. Key dependencies:
- PyTorch >= 2.0.0
- librosa >= 0.10.0
- scikit-learn >= 1.3.0
- sentence-transformers >= 2.2.0

## License

This project is for educational purposes.

## Author

404sanchita
