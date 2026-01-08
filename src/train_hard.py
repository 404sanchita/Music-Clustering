"""
Hard Task: Conditional VAE with audio + text + labels
- Train Conditional VAE (CVAE)
- Compare with Autoencoder baseline and Direct Spectral baseline
- Apply K-Means clustering
- Calculate comprehensive metrics (including NMI, Cluster Purity)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from model import HardMusicCVAE, SimpleAutoencoder
from clustering import kmeans_clustering
from evaluation import calculate_all_metrics, print_metrics
import pandas as pd

def train_cvae(model, dataloader, epochs=50, lr=0.001, device='cpu'):
    """Train Conditional VAE"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            audio = batch[0].to(device)
            text = batch[1].to(device)
            labels_onehot = batch[2].to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(audio, text, labels_onehot)
            recon_loss = F.mse_loss(recon, audio, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader.dataset):.4f}")
    
    return model

def train_autoencoder(model, dataloader, epochs=50, lr=0.001, device='cpu'):
    """Train Autoencoder baseline"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            audio = batch[0].to(device)
            text = batch[1].to(device)
            
            optimizer.zero_grad()
            recon, latent = model(audio, text)
            loss = F.mse_loss(recon, audio, reduction='sum')
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader.dataset):.4f}")
    
    return model

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load hybrid data
    if os.path.exists('hybrid_data.pt'):
        print("Loading hybrid data...")
        data = torch.load('hybrid_data.pt')
        spectrograms = data['spectrograms']
        text_embeddings = data['text_embeddings']
        labels = data['labels']
    else:
        print("Error: hybrid_data.pt not found. Run hybrid_data.py first.")
        exit(1)
    
    print(f"Spectrograms shape: {spectrograms.shape}")
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Create one-hot encoded labels
    num_classes = 6
    labels_onehot = F.one_hot(labels, num_classes=num_classes).float()
    true_labels = labels.numpy()
    
    # 2. Train Conditional VAE
    print("\n" + "="*60)
    print("Training Conditional VAE (CVAE)")
    print("="*60)
    dataset_cvae = TensorDataset(spectrograms, text_embeddings, labels_onehot)
    dataloader_cvae = DataLoader(dataset_cvae, batch_size=16, shuffle=True)
    
    cvae_model = HardMusicCVAE(latent_dim=32, text_dim=text_embeddings.shape[1], num_classes=num_classes).to(device)
    cvae_model = train_cvae(cvae_model, dataloader_cvae, epochs=50, device=device)
    
    # Extract CVAE latent representations
    cvae_model.eval()
    all_audio = spectrograms.to(device)
    all_text = text_embeddings.to(device)
    all_labels_onehot = labels_onehot.to(device)
    
    with torch.no_grad():
        mu, _ = cvae_model.encode(all_audio, all_text, all_labels_onehot)
        latent_cvae = mu.cpu().numpy()
    
    print(f"CVAE Latent shape: {latent_cvae.shape}")
    
    # 3. Train Autoencoder Baseline
    print("\n" + "="*60)
    print("Training Autoencoder Baseline")
    print("="*60)
    dataset_ae = TensorDataset(spectrograms, text_embeddings)
    dataloader_ae = DataLoader(dataset_ae, batch_size=16, shuffle=True)
    
    ae_model = SimpleAutoencoder(latent_dim=32, text_dim=text_embeddings.shape[1]).to(device)
    ae_model = train_autoencoder(ae_model, dataloader_ae, epochs=50, device=device)
    
    # Extract Autoencoder latent representations
    ae_model.eval()
    with torch.no_grad():
        _, latent_ae = ae_model(all_audio, all_text)
        latent_ae_np = latent_ae.cpu().numpy()
    
    print(f"Autoencoder Latent shape: {latent_ae_np.shape}")
    
    # 4. Direct Spectral Baseline (flatten spectrograms)
    print("\n" + "="*60)
    print("Direct Spectral Baseline")
    print("="*60)
    spectral_features = spectrograms.view(spectrograms.shape[0], -1).numpy()
    pca_spectral = PCA(n_components=32)
    spectral_pca = pca_spectral.fit_transform(spectral_features)
    print(f"Direct Spectral PCA shape: {spectral_pca.shape}")
    
    # 5. Clustering & Evaluation
    print("\n" + "="*60)
    print("Applying K-Means Clustering")
    print("="*60)
    
    n_clusters = 6
    all_metrics = {}
    
    # CVAE + K-Means
    print("\nCVAE + K-Means:")
    labels_cvae, _ = kmeans_clustering(latent_cvae, n_clusters=n_clusters)
    metrics_cvae = calculate_all_metrics(latent_cvae, labels_cvae, true_labels)
    all_metrics['cvae_kmeans'] = metrics_cvae
    print_metrics(metrics_cvae, "CVAE + K-Means")
    
    # Autoencoder + K-Means
    print("\nAutoencoder + K-Means:")
    labels_ae, _ = kmeans_clustering(latent_ae_np, n_clusters=n_clusters)
    metrics_ae = calculate_all_metrics(latent_ae_np, labels_ae, true_labels)
    all_metrics['autoencoder_kmeans'] = metrics_ae
    print_metrics(metrics_ae, "Autoencoder + K-Means")
    
    # Direct Spectral + K-Means
    print("\nDirect Spectral + K-Means:")
    labels_spectral, _ = kmeans_clustering(spectral_pca, n_clusters=n_clusters)
    metrics_spectral = calculate_all_metrics(spectral_pca, labels_spectral, true_labels)
    all_metrics['direct_spectral_kmeans'] = metrics_spectral
    print_metrics(metrics_spectral, "Direct Spectral + K-Means")
    
    # 6. Save metrics to CSV
    print("\n" + "="*60)
    print("Saving Metrics")
    print("="*60)
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')
    metrics_df['method'] = metrics_df.index
    metrics_df['task'] = 'hard'
    metrics_df['algorithm'] = 'kmeans'
    
    # Reorder columns
    metrics_df = metrics_df[['method', 'task', 'silhouette_score', 'calinski_harabasz_index', 
                             'davies_bouldin_index', 'adjusted_rand_index', 'nmi', 'cluster_purity', 'algorithm']]
    
    csv_path = os.path.join('results', 'clustering_metrics.csv')
    os.makedirs('results', exist_ok=True)
    
    if not os.path.exists(csv_path):
        metrics_df.to_csv(csv_path, index=False)
    else:
        metrics_df.to_csv(csv_path, mode='a', header=False, index=False)
    print(f"Metrics saved to {csv_path}")
    
    # 7. Visualization
    print("\n" + "="*60)
    print("Creating Visualizations")
    print("="*60)
    os.makedirs('results/latent_visualization', exist_ok=True)
    
    # t-SNE for all methods
    tsne = TSNE(n_components=2, random_state=42)
    
    latent_cvae_2d = tsne.fit_transform(latent_cvae)
    latent_ae_2d = tsne.fit_transform(latent_ae_np)
    spectral_2d = tsne.fit_transform(spectral_pca)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # True labels (CVAE space)
    scatter = axes[0, 0].scatter(latent_cvae_2d[:, 0], latent_cvae_2d[:, 1], c=true_labels, cmap='tab10', alpha=0.6)
    axes[0, 0].set_title('True Labels (CVAE Latent Space)')
    axes[0, 0].set_xlabel('t-SNE 1')
    axes[0, 0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # CVAE + K-Means
    scatter = axes[0, 1].scatter(latent_cvae_2d[:, 0], latent_cvae_2d[:, 1], c=labels_cvae, cmap='tab10', alpha=0.6)
    axes[0, 1].set_title('CVAE + K-Means')
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=axes[0, 1])
    
    # Autoencoder + K-Means
    scatter = axes[1, 0].scatter(latent_ae_2d[:, 0], latent_ae_2d[:, 1], c=labels_ae, cmap='tab10', alpha=0.6)
    axes[1, 0].set_title('Autoencoder + K-Means')
    axes[1, 0].set_xlabel('t-SNE 1')
    axes[1, 0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=axes[1, 0])
    
    # Direct Spectral + K-Means
    scatter = axes[1, 1].scatter(spectral_2d[:, 0], spectral_2d[:, 1], c=labels_spectral, cmap='tab10', alpha=0.6)
    axes[1, 1].set_title('Direct Spectral + K-Means')
    axes[1, 1].set_xlabel('t-SNE 1')
    axes[1, 1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('results/latent_visualization/hard_latent_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to results/latent_visualization/hard_latent_comparison.png")
    plt.close()
    
    print("\n" + "="*60)
    print("Hard Task Complete!")
    print("="*60)

