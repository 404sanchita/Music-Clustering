"""
Easy Task: Basic VAE with MFCC features
- Extract MFCC features
- Train basic VAE
- Apply K-Means clustering
- Compare with PCA baseline
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score
from model import MusicVAE
from preprocess import extract_features
from clustering import kmeans_clustering
from evaluation import calculate_all_metrics, print_metrics

def train_vae(model, dataloader, epochs=50, lr=0.001, device='cpu'):
    """Train VAE model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            
            recon, mu, logvar = model(x)
            recon_loss = F.mse_loss(recon, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader.dataset):.4f}")
    
    return model

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load or extract features
    if os.path.exists('processed_data.pt'):
        print("Loading processed data...")
        data = torch.load('processed_data.pt')
        features = data['features']
        labels = data['labels']
    else:
        print("Extracting features...")
        features, labels = extract_features('dataset/audio')
        torch.save({'features': features, 'labels': labels}, 'processed_data.pt')
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # 2. Prepare data
    dataset = TensorDataset(features)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 3. Train VAE
    print("\n" + "="*60)
    print("Training VAE")
    print("="*60)
    model = MusicVAE(input_dim=features.shape[1], latent_dim=32).to(device)
    model = train_vae(model, dataloader, epochs=50, device=device)
    
    # 4. Extract latent representations
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(features.to(device))
        latent = mu.cpu().numpy()
    
    print(f"Latent shape: {latent.shape}")
    
    # 5. Clustering on VAE latent space
    print("\n" + "="*60)
    print("K-Means Clustering on VAE Latent Space")
    print("="*60)
    labels_kmeans, _ = kmeans_clustering(latent, n_clusters=6)
    metrics_vae = calculate_all_metrics(latent, labels_kmeans, labels.numpy())
    print_metrics(metrics_vae, "VAE + K-Means")
    
    # 6. PCA Baseline
    print("\n" + "="*60)
    print("PCA Baseline")
    print("="*60)
    pca = PCA(n_components=32)
    features_pca = pca.fit_transform(features.numpy())
    labels_pca, _ = kmeans_clustering(features_pca, n_clusters=6)
    metrics_pca = calculate_all_metrics(features_pca, labels_pca, labels.numpy())
    print_metrics(metrics_pca, "PCA + K-Means")
    
    # 7. Visualization
    print("\n" + "="*60)
    print("Creating Visualizations")
    print("="*60)
    os.makedirs('results/latent_visualization', exist_ok=True)
    
    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # True labels
    scatter1 = axes[0].scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels.numpy(), cmap='tab10', alpha=0.6)
    axes[0].set_title('True Labels (t-SNE of VAE Latent Space)')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter1, ax=axes[0])
    
    # Predicted clusters
    scatter2 = axes[1].scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels_kmeans, cmap='tab10', alpha=0.6)
    axes[1].set_title('K-Means Clusters (t-SNE of VAE Latent Space)')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('results/latent_visualization/easy_task_results.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to results/latent_visualization/easy_task_results.png")
    plt.close()
    
    print("\n" + "="*60)
    print("Easy Task Complete!")
    print("="*60)

