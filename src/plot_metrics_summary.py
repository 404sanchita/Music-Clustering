"""
Plot summary of all clustering metrics across tasks and methods
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def plot_metrics_table(df, save_path):
    """Create a table visualization of metrics"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = df[['method', 'task', 'silhouette_score', 'calinski_harabasz_index', 
                     'davies_bouldin_index', 'adjusted_rand_index', 'nmi', 'cluster_purity']].copy()
    
    # Round numeric columns
    numeric_cols = ['silhouette_score', 'calinski_harabasz_index', 'davies_bouldin_index', 
                    'adjusted_rand_index', 'nmi', 'cluster_purity']
    for col in numeric_cols:
        table_data[col] = table_data[col].round(4)
    
    # Create table
    table = ax.table(cellText=table_data.values, colLabels=table_data.columns,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Clustering Metrics Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_bar_charts(df, save_path):
    """Create bar charts for each metric"""
    metrics = ['silhouette_score', 'calinski_harabasz_index', 'davies_bouldin_index', 
               'adjusted_rand_index', 'nmi', 'cluster_purity']
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        data = df.groupby(['task', 'method'])[metric].mean().unstack(fill_value=0)
        
        x = np.arange(len(data.index))
        width = 0.8 / len(data.columns)
        
        for i, col in enumerate(data.columns):
            offset = (i - len(data.columns)/2 + 0.5) * width
            ax.bar(x + offset, data[col], width, label=col.replace('_', ' ').title())
        
        ax.set_xlabel('Task')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xticks(x)
        ax.set_xticklabels(data.index)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    metrics_csv_path = project_root / 'results' / 'clustering_metrics.csv'
    viz_dir = project_root / 'results' / 'latent_visualization'
    os.makedirs(viz_dir, exist_ok=True)
    
    if not metrics_csv_path.exists():
        print(f"Error: Metrics CSV not found at {metrics_csv_path}")
        print("Please run train_medium.py and train_hard.py first to generate metrics.")
    else:
        metrics_df = pd.read_csv(metrics_csv_path)
        
        # Generate and save table plot
        plot_metrics_table(metrics_df, viz_dir / 'metrics_table.png')
        print(f"Saved metrics table to: {viz_dir / 'metrics_table.png'}")
        
        # Generate and save bar charts
        plot_metrics_bar_charts(metrics_df, viz_dir / 'metrics_bars.png')
        print(f"Saved metrics bar charts to: {viz_dir / 'metrics_bars.png'}")

