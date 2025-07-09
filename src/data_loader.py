import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

class MultiModalDataset(Dataset):
    """Dataset class for multimodal sentiment analysis"""
    
    def __init__(self, csv_path, features_dir):
        self.csv_path = Path(csv_path)
        self.features_dir = Path(features_dir)
        
        # Load metadata
        self.data = pd.read_csv(self.csv_path)
        
        # Convert sentiment labels from -1,0,1 to 0,1,2 for classification
        self.data['sentiment_class'] = self.data['sentiment_label'] + 1
        
        logging.info(f"📁 Dataset loaded: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_id = row['video_id']
        text = row['text']
        sentiment_label = row['sentiment_class']  # 0, 1, 2
        
        # Load pre-saved features (or generate dummy ones)
        text_features = self._load_feature('text', video_id)
        audio_features = self._load_feature('audio', video_id)
        visual_features = self._load_feature('visual', video_id)
        
        return {
            'text': text,  # Raw text for tokenization
            'text_features': text_features,  # Pre-extracted features (if available)
            'audio_features': audio_features,
            'visual_features': visual_features,
            'sentiment_label': torch.tensor(sentiment_label, dtype=torch.long),
            'video_id': video_id
        }
    
    def _load_feature(self, modality, video_id):
        """Load pre-saved feature from file"""
        feature_path = self.features_dir / modality / f"{video_id}.npy"
        if feature_path.exists():
            feature = np.load(feature_path)
        else:
            # Generate dummy feature if file missing
            feature = np.random.randn(768)
        return torch.FloatTensor(feature)


def create_data_loaders(csv_path, features_dir, batch_size=4):
    """
    Create train, validation, and test data loaders
    
    Args:
        csv_path: Path to CSV file with metadata
        features_dir: Directory containing feature files
        batch_size: Batch size for data loaders
    """
    # Load dataset
    dataset = MultiModalDataset(csv_path, features_dir)
    
    # Create train/val/test splits
    indices = list(range(len(dataset)))
    train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def collate_fn(batch):
    """
    Custom collate function to handle text tokenization
    """
    texts = [item['text'] for item in batch]
    text_features = torch.stack([item['text_features'] for item in batch])
    audio_features = torch.stack([item['audio_features'] for item in batch])
    visual_features = torch.stack([item['visual_features'] for item in batch])
    labels = torch.stack([item['sentiment_label'] for item in batch])
    video_ids = [item['video_id'] for item in batch]
    
    return {
        'texts': texts,  # List of raw texts
        'text_features': text_features,
        'audio_features': audio_features,
        'visual_features': visual_features,
        'labels': labels,
        'video_ids': video_ids
    }

def get_sample_data():
    """Get a single sample for testing"""
    csv_path = "data/sample_multimodal/sample_multimodal_data.csv"
    features_dir = "data/sample_multimodal/features"
    
    dataset = MultiModalDataset(csv_path, features_dir, split='train')
    return dataset[0]
