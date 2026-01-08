import os
import librosa
import numpy as np
import torch

def extract_features(base_path, n_mfcc=13, duration=30):
    """
    Extract MFCC features from audio files in language subfolders.
    
    Args:
        base_path: Path to dataset/audio directory
        n_mfcc: Number of MFCC coefficients
        duration: Maximum duration to load (seconds)
    
    Returns:
        features: Tensor of MFCC features [N, n_mfcc]
        labels: Tensor of language labels [N]
    """
    features = []
    labels = []
    lang_map = {'bangla': 0, 'english': 1, 'hindi': 2, 'korean': 3, 'japanese': 4, 'spanish': 5}
    
    for lang_name, lang_id in lang_map.items():
        folder_path = os.path.join(base_path, lang_name)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} not found, skipping...")
            continue
        
        print(f"Processing: {lang_name}")
        count = 0
        for file in os.listdir(folder_path):
            if file.endswith(('.mp3', '.wav')):
                try:
                    path = os.path.join(folder_path, file)
                    y, sr = librosa.load(path, duration=duration)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                    mfcc_mean = np.mean(mfcc.T, axis=0)
                    features.append(mfcc_mean)
                    labels.append(lang_id)
                    count += 1
                except Exception as e:
                    print(f"Error processing {file}: {e}")
        print(f"  Processed {count} files")
    
    if len(features) == 0:
        raise ValueError("No songs found. Check your folder path!")
    
    print(f"\nTotal files processed: {len(features)}")
    return torch.tensor(np.array(features), dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

if __name__ == "__main__":
    # Example usage
    base_path = "dataset/audio"
    if os.path.exists(base_path):
        features, labels = extract_features(base_path)
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Save processed data
        torch.save({'features': features, 'labels': labels}, 'processed_data.pt')
        print("Saved to processed_data.pt")
    else:
        print(f"Error: {base_path} not found!")

