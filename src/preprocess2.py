import os
import librosa
import numpy as np
import torch

def extract_spectrograms(base_path, n_mels=128, duration=30, target_length=128):
    """
    Extract mel spectrograms from audio files in language subfolders.
    
    Args:
        base_path: Path to dataset/audio directory
        n_mels: Number of mel filter banks
        duration: Maximum duration to load (seconds)
        target_length: Target time frames (will pad or truncate)
    
    Returns:
        spectrograms: Tensor of spectrograms [N, 1, n_mels, target_length]
        labels: Tensor of language labels [N]
    """
    spectrograms = []
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
                    
                    # Compute mel spectrogram
                    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    # Normalize to [0, 1]
                    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
                    
                    # Pad or truncate to target_length
                    if mel_spec_db.shape[1] > target_length:
                        mel_spec_db = mel_spec_db[:, :target_length]
                    elif mel_spec_db.shape[1] < target_length:
                        pad_width = target_length - mel_spec_db.shape[1]
                        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
                    
                    # Add channel dimension: [1, n_mels, target_length]
                    mel_spec_db = mel_spec_db[np.newaxis, :, :]
                    spectrograms.append(mel_spec_db)
                    labels.append(lang_id)
                    count += 1
                except Exception as e:
                    print(f"Error processing {file}: {e}")
        print(f"  Processed {count} files")
    
    if len(spectrograms) == 0:
        raise ValueError("No songs found. Check your folder path!")
    
    print(f"\nTotal files processed: {len(spectrograms)}")
    spectrograms_tensor = torch.tensor(np.array(spectrograms), dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    print(f"Spectrograms shape: {spectrograms_tensor.shape}")
    print(f"Labels shape: {labels_tensor.shape}")
    
    return spectrograms_tensor, labels_tensor

if __name__ == "__main__":
    # Example usage
    base_path = "dataset/audio"
    if os.path.exists(base_path):
        spectrograms, labels = extract_spectrograms(base_path)
        
        # Save processed data
        torch.save({
            'spectrograms': spectrograms,
            'labels': labels
        }, 'processed_data_2d_labeled.pt')
        print("Saved to processed_data_2d_labeled.pt")
    else:
        print(f"Error: {base_path} not found!")

