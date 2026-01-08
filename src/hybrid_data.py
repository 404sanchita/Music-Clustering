import torch
import numpy as np
from sentence_transformers import SentenceTransformer

def create_hybrid_data(spectrograms, labels, lang_names=None):
    """
    Create hybrid data combining audio spectrograms with text embeddings.
    
    Args:
        spectrograms: Tensor of audio spectrograms [N, 1, H, W]
        labels: Tensor of language labels [N]
        lang_names: Optional list of language names for text embedding
    
    Returns:
        Dictionary with spectrograms, text_embeddings, and labels
    """
    # Language names mapping
    lang_map = {0: 'bangla', 1: 'english', 2: 'hindi', 3: 'korean', 4: 'japanese', 5: 'spanish'}
    
    if lang_names is None:
        lang_names = [lang_map[int(label.item())] for label in labels]
    
    # Load sentence transformer model
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate text embeddings
    print("Generating text embeddings...")
    text_embeddings = model.encode(lang_names, convert_to_tensor=True)
    
    print(f"Spectrograms shape: {spectrograms.shape}")
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return {
        'spectrograms': spectrograms,
        'text_embeddings': text_embeddings,
        'labels': labels
    }

if __name__ == "__main__":
    # Example usage
    try:
        # Load processed spectrograms
        data = torch.load('processed_data_2d_labeled.pt')
        spectrograms = data['spectrograms']
        labels = data['labels']
        
        # Create hybrid data
        hybrid_data = create_hybrid_data(spectrograms, labels)
        
        # Save hybrid data
        torch.save(hybrid_data, 'hybrid_data.pt')
        print("Saved hybrid data to hybrid_data.pt")
    except FileNotFoundError:
        print("Error: processed_data_2d_labeled.pt not found. Run preprocess2.py first.")

