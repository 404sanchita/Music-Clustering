"""
Medium Task: Hybrid ConvVAE with audio + text
- Use spectrograms and text embeddings
- Train HybridConvVAE
- Apply multiple clustering algorithms (K-Means, Agglomerative, DBSCAN)
- Calculate comprehensive metrics
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.manifold import TSNE
from model import HybridConvVAE
from clustering import apply_all_clustering_algorithms
from evaluation import calculate_all_metrics, print_metrics
import pandas as pd

def train_vae(model, dataloader, epochs=50, lr=0.001, device='cpu'):
    """Train HybridConvVAE model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            audio = batch[0].to(device)
            text = batch[1].to(device)
            optimizer.zero_grad()
            
            recon, mu, logvar = model(audio, text)
            recon_loss = F.mse_loss(recon, audio, reduction='sum')
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
    
    # 2. Prepare data
    dataset = TensorDataset(spectrograms, text_embeddings)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 3. Train HybridConvVAE
    print("\n" + "="*60)
    print("Training HybridConvVAE")
    print("="*60)
    model = HybridConvVAE(latent_dim=32, text_dim=text_embeddings.shape[1]).to(device)
    model = train_vae(model, dataloader, epochs=50, device=device)
    
    # 4. Extract latent representations
    print("\n" + "="*60)
    print("Extracting Latent Representations")
    print("="*60)
    model.eval()
    all_audio = spectrograms.to(device)
    all_text = text_embeddings.to(device)
    
    with torch.no_grad():
        mu, _ = model.encode(all_audio, all_text)
        latent_np = mu.cpu().numpy()
    
    print(f"Latent shape: {latent_np.shape}")
    
    # 5. Clustering & Evaluation
    print("\n" + "="*60)
    print("Applying Clustering Algorithms")
    print("="*60)
    
    n_clusters = 6
    true_labels = labels.numpy()
    clustering_results = apply_all_clustering_algorithms(latent_np, n_clusters=n_clusters, standardize=True)
    
    all_metrics = {}
    for algo_name, result in clustering_results.items():
        print(f"\n{algo_name.replace('_', ' ').title()} Clustering:")
        metrics = calculate_all_metrics(
            latent_np,
            result['labels'],
            true_labels
        )
        all_metrics[algo_name] = metrics
        print_metrics(metrics, title=f"{algo_name.replace('_', ' ').title()} Metrics")
    
    # 6. Save metrics to CSV
    print("\n" + "="*60)
    print("Saving Metrics")
    print("="*60)
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')
    metrics_df['method'] = metrics_df.index
    metrics_df['task'] = 'medium'
    metrics_df['algorithm'] = metrics_df.index
    
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
    
    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_np)
    
    n_algorithms = len(clustering_results)
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    # True labels
    scatter = axes[0].scatter(latent_2d[:, 0], latent_2d[:, 1], c=true_labels, cmap='tab10', alpha=0.6)
    axes[0].set_title('True Labels')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=axes[0])
    
    # Clustering results
    for idx, (algo_name, result) in enumerate(clustering_results.items(), 1):
        if idx >= len(axes):
            break
        scatter = axes[idx].scatter(latent_2d[:, 0], latent_2d[:, 1], c=result['labels'], cmap='tab10', alpha=0.6)
        axes[idx].set_title(f'{algo_name.replace("_", " ").title()} Clustering')
        axes[idx].set_xlabel('t-SNE 1')
        axes[idx].set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=axes[idx])
    
    plt.tight_layout()
    plt.savefig('results/latent_visualization/medium_clustering_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to results/latent_visualization/medium_clustering_comparison.png")
    plt.close()
    
    print("\n" + "="*60)
    print("Medium Task Complete!")
    print("="*60)

