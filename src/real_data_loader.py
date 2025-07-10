import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import librosa
from PIL import Image
import logging
from sklearn.model_selection import train_test_split

class TextSentimentDataset(Dataset):
    """Dataset for text sentiment analysis using SST-2"""
    
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.data = pd.read_csv(self.csv_path)
        
        logging.info(f"📁 Text dataset loaded: {len(self.data)} samples")
        logging.info(f"   Label distribution: {self.data['label'].value_counts().to_dict()}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['text']
        label = row['label']  # 0 for negative, 1 for positive
        
        return {
            'text': text,
            'label': torch.tensor(label, dtype=torch.long),
            'sample_id': f"text_{idx}"
        }


class AudioEmotionDataset(Dataset):
    """Dataset for audio emotion recognition using EMO-DB"""
    
    def __init__(self, labels_csv_path, audio_dir):
        self.labels_csv_path = Path(labels_csv_path)
        self.audio_dir = Path(audio_dir)
        self.data = pd.read_csv(self.labels_csv_path)
        
        # Emotion mapping to sentiment (simplified)
        # EMO-DB emotions: happiness, neutral, anger, sadness, fear, boredom, disgust
        self.emotion_to_sentiment = {
            'neutral': 1,     # neutral -> positive
            'happiness': 1,   # happiness -> positive
            'boredom': 1,     # boredom -> positive (neutral-ish)
            'sadness': 0,     # sadness -> negative
            'anger': 0,       # anger -> negative
            'fear': 0,        # fear -> negative
            'disgust': 0      # disgust -> negative
        }
        
        # Map emotions to sentiment labels
        self.data['sentiment'] = self.data['emotion'].map(self.emotion_to_sentiment)
        
        # Check for unmapped emotions
        unmapped = self.data[self.data['sentiment'].isna()]
        if len(unmapped) > 0:
            logging.warning(f"Found {len(unmapped)} samples with unmapped emotions:")
            logging.warning(f"   Unmapped emotions: {unmapped['emotion'].unique().tolist()}")
            # Remove unmapped samples
            self.data = self.data.dropna(subset=['sentiment'])
            logging.warning(f"   Removed unmapped samples, remaining: {len(self.data)} samples")
        
        # Ensure sentiment labels are integers
        self.data['sentiment'] = self.data['sentiment'].astype(int)
        
        logging.info(f"📁 Audio dataset loaded: {len(self.data)} samples")
        logging.info(f"   Emotion distribution: {self.data['emotion'].value_counts().to_dict()}")
        logging.info(f"   Sentiment distribution: {self.data['sentiment'].value_counts().to_dict()}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row['filename']
        emotion = row['emotion']
        sentiment = row['sentiment']
        
        # Load audio file
        audio_path = self.audio_dir / filename
        if audio_path.exists():
            try:
                # Load audio with librosa
                audio, sr = librosa.load(str(audio_path), sr=16000)
                # Convert to tensor and pad/truncate to fixed length (3 seconds)
                target_length = 3 * sr  # 3 seconds
                if len(audio) > target_length:
                    audio = audio[:target_length]
                else:
                    audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
                audio_tensor = torch.FloatTensor(audio)
            except Exception as e:
                logging.warning(f"Error loading audio {audio_path}: {e}")
                # Create dummy audio if loading fails
                audio_tensor = torch.randn(3 * 16000)
        else:
            logging.warning(f"Audio file not found: {audio_path}")
            # Create dummy audio if file missing
            audio_tensor = torch.randn(3 * 16000)
        
        return {
            'audio': audio_tensor,
            'emotion': emotion,
            'label': torch.tensor(sentiment, dtype=torch.long),
            'sample_id': f"audio_{filename}"
        }


class ImageEmotionDataset(Dataset):
    """Dataset for image emotion recognition using FER-2013"""
    
    def __init__(self, labels_csv_path, image_dirs):
        self.labels_csv_path = Path(labels_csv_path)
        self.image_dirs = [Path(d) for d in image_dirs]  # List of train, test dirs
        self.data = pd.read_csv(self.labels_csv_path)
        
        # Emotion mapping to sentiment (simplified)
        self.emotion_to_sentiment = {
            'neutral': 1,     # neutral -> positive
            'happy': 1,       # happy -> positive
            'surprise': 1,    # surprise -> positive
            'sad': 0,         # sad -> negative
            'angry': 0,       # angry -> negative
            'fear': 0,        # fear -> negative
            'disgust': 0      # disgust -> negative
        }
        
        # Map emotions to sentiment labels
        self.data['sentiment'] = self.data['emotion'].map(self.emotion_to_sentiment)
        
        logging.info(f"📁 Image dataset loaded: {len(self.data)} samples")
        logging.info(f"   Emotion distribution: {self.data['emotion'].value_counts().to_dict()}")
        logging.info(f"   Sentiment distribution: {self.data['sentiment'].value_counts().to_dict()}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row['filename']
        emotion = row['emotion']
        sentiment = row['sentiment']
        
        # Use the path column if available, otherwise construct from emotion and filename
        if 'path' in row and pd.notna(row['path']):
            relative_path = row['path']
        else:
            # Fallback: construct path from emotion and split
            split = row.get('split', 'train')
            relative_path = f"{split}/{emotion}/{filename}"
        
        # Try to find image using the relative path
        image_path = None
        base_dir = Path(self.labels_csv_path).parent  # Get the fer-2013 directory
        potential_path = base_dir / relative_path
        
        if potential_path.exists():
            image_path = potential_path
        else:
            # Fallback: try the old method (direct in train/test dirs)
            for img_dir in self.image_dirs:
                potential_path = img_dir / filename
                if potential_path.exists():
                    image_path = potential_path
                    break
        
        if image_path and image_path.exists():
            try:
                # Load image
                image = Image.open(image_path).convert('RGB')
                # Convert to tensor (simple conversion, could be improved with transforms)
                image = np.array(image)
                if len(image.shape) == 2:  # Grayscale
                    image = np.stack([image] * 3, axis=-1)  # Convert to RGB
                # Resize to standard size
                image = cv2.resize(image, (224, 224))
                # Normalize to [0, 1]
                image = image.astype(np.float32) / 255.0
                # Convert to tensor (C, H, W)
                image_tensor = torch.FloatTensor(image).permute(2, 0, 1)
            except Exception as e:
                logging.warning(f"Error loading image {image_path}: {e}")
                # Create dummy image if loading fails
                image_tensor = torch.randn(3, 224, 224)
        else:
            logging.warning(f"Image file not found: {filename}")
            # Create dummy image if file missing
            image_tensor = torch.randn(3, 224, 224)
        
        return {
            'image': image_tensor,
            'emotion': emotion,
            'label': torch.tensor(sentiment, dtype=torch.long),
            'sample_id': f"image_{filename}"
        }


class MultiModalRealDataset(Dataset):
    """Combined multimodal dataset using real data from all modalities"""
    
    def __init__(self, text_csv, audio_csv, audio_dir, image_csv, image_dirs, 
                 balance_modalities=True, max_samples_per_modality=1000):
        self.text_dataset = TextSentimentDataset(text_csv)
        self.audio_dataset = AudioEmotionDataset(audio_csv, audio_dir)
        self.image_dataset = ImageEmotionDataset(image_csv, image_dirs)
        
        # If balancing, limit each modality to max_samples_per_modality
        if balance_modalities:
            text_indices = list(range(min(len(self.text_dataset), max_samples_per_modality)))
            audio_indices = list(range(min(len(self.audio_dataset), max_samples_per_modality)))
            image_indices = list(range(min(len(self.image_dataset), max_samples_per_modality)))
        else:
            text_indices = list(range(len(self.text_dataset)))
            audio_indices = list(range(len(self.audio_dataset)))
            image_indices = list(range(len(self.image_dataset)))
        
        # Create mapping: (modality, index_in_modality)
        self.samples = []
        self.samples.extend([('text', i) for i in text_indices])
        self.samples.extend([('audio', i) for i in audio_indices])
        self.samples.extend([('image', i) for i in image_indices])
        
        logging.info(f"📁 Multimodal dataset created: {len(self.samples)} total samples")
        logging.info(f"   Text samples: {len(text_indices)}")
        logging.info(f"   Audio samples: {len(audio_indices)}")
        logging.info(f"   Image samples: {len(image_indices)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        modality, mod_idx = self.samples[idx]
        
        if modality == 'text':
            sample = self.text_dataset[mod_idx]
            return {
                'text': sample['text'],
                'audio': torch.zeros(3 * 16000),  # Dummy audio
                'image': torch.zeros(3, 224, 224),  # Dummy image
                'label': sample['label'],
                'modality': 'text',
                'sample_id': sample['sample_id']
            }
        elif modality == 'audio':
            sample = self.audio_dataset[mod_idx]
            return {
                'text': "",  # Empty text
                'audio': sample['audio'],
                'image': torch.zeros(3, 224, 224),  # Dummy image
                'label': sample['label'],
                'modality': 'audio',
                'sample_id': sample['sample_id']
            }
        elif modality == 'image':
            sample = self.image_dataset[mod_idx]
            return {
                'text': "",  # Empty text
                'audio': torch.zeros(3 * 16000),  # Dummy audio
                'image': sample['image'],
                'label': sample['label'],
                'modality': 'image',
                'sample_id': sample['sample_id']
            }


def create_real_data_loaders(data_config, batch_size=8, balance_modalities=True, max_samples_per_modality=1000):
    """
    Create data loaders for real datasets
    
    Args:
        data_config: Dictionary with paths to datasets
        batch_size: Batch size for data loaders
        balance_modalities: Whether to balance samples across modalities
        max_samples_per_modality: Maximum samples per modality if balancing
    """
    
    # Create combined dataset
    dataset = MultiModalRealDataset(
        text_csv=data_config['text']['train_csv'],
        audio_csv=data_config['audio']['labels_csv'],
        audio_dir=data_config['audio']['audio_dir'],
        image_csv=data_config['image']['labels_csv'],
        image_dirs=data_config['image']['image_dirs'],
        balance_modalities=balance_modalities,
        max_samples_per_modality=max_samples_per_modality
    )
    
    # Create train/val/test splits
    indices = list(range(len(dataset)))
    train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=multimodal_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=multimodal_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=multimodal_collate_fn)
    
    logging.info(f"📊 Data loaders created:")
    logging.info(f"   Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    logging.info(f"   Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    logging.info(f"   Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader


def create_single_modality_loaders(modality, data_config, batch_size=8):
    """Create data loaders for a single modality"""
    
    if modality == 'text':
        # Use train/dev/test splits from SST-2
        train_dataset = TextSentimentDataset(data_config['text']['train_csv'])
        val_dataset = TextSentimentDataset(data_config['text']['dev_csv'])
        test_dataset = TextSentimentDataset(data_config['text']['test_csv'])
        
        collate_func = text_collate_fn
        
    elif modality == 'audio':
        # Use all audio data and split it
        full_dataset = AudioEmotionDataset(
            data_config['audio']['labels_csv'],
            data_config['audio']['audio_dir']
        )
        
        indices = list(range(len(full_dataset)))
        train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
        val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
        
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        
        collate_func = audio_collate_fn
        
    elif modality == 'image':
        # Use all image data and split it
        full_dataset = ImageEmotionDataset(
            data_config['image']['labels_csv'],
            data_config['image']['image_dirs']
        )
        
        indices = list(range(len(full_dataset)))
        train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
        val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
        
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        
        collate_func = image_collate_fn
    
    else:
        raise ValueError(f"Unknown modality: {modality}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_func)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_func)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_func)
    
    return train_loader, val_loader, test_loader


def text_collate_fn(batch):
    """Collate function for text-only batches"""
    texts = [item['text'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    sample_ids = [item['sample_id'] for item in batch]
    
    return {
        'texts': texts,
        'labels': labels,
        'sample_ids': sample_ids
    }


def audio_collate_fn(batch):
    """Collate function for audio-only batches"""
    audios = torch.stack([item['audio'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    sample_ids = [item['sample_id'] for item in batch]
    
    return {
        'audios': audios,
        'labels': labels,
        'sample_ids': sample_ids
    }


def image_collate_fn(batch):
    """Collate function for image-only batches"""
    images = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    sample_ids = [item['sample_id'] for item in batch]
    
    return {
        'images': images,
        'labels': labels,
        'sample_ids': sample_ids
    }


def multimodal_collate_fn(batch):
    """Collate function for multimodal batches"""
    texts = [item['text'] for item in batch]
    audios = torch.stack([item['audio'] for item in batch])
    images = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    modalities = [item['modality'] for item in batch]
    sample_ids = [item['sample_id'] for item in batch]
    
    return {
        'texts': texts,
        'audios': audios,
        'images': images,
        'labels': labels,
        'modalities': modalities,
        'sample_ids': sample_ids
    }


def get_data_config():
    """Get default data configuration for real datasets"""
    return {
        'text': {
            'train_csv': 'data/text_sentiment/sst2/train.csv',
            'dev_csv': 'data/text_sentiment/sst2/dev.csv',
            'test_csv': 'data/text_sentiment/sst2/test.csv'
        },
        'audio': {
            'labels_csv': 'data/audio_emotion/emodb/labels.csv',
            'audio_dir': 'data/audio_emotion/emodb/wav'
        },
        'image': {
            'labels_csv': 'data/image_emotion/fer-2013/labels.csv',
            'image_dirs': [
                'data/image_emotion/fer-2013/train',
                'data/image_emotion/fer-2013/test'
            ]
        }
    }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Get data configuration
    data_config = get_data_config()
    
    # Test text dataset
    print("Testing text dataset...")
    text_dataset = TextSentimentDataset(data_config['text']['train_csv'])
    sample = text_dataset[0]
    print(f"Text sample: {sample}")
    
    # Create multimodal data loaders
    print("\nCreating multimodal data loaders...")
    train_loader, val_loader, test_loader = create_real_data_loaders(
        data_config, 
        batch_size=4, 
        balance_modalities=True, 
        max_samples_per_modality=100  # Small for testing
    )
    
    # Test a batch
    print("\nTesting a batch...")
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Batch sizes: texts={len(batch['texts'])}, audios={batch['audios'].shape}, images={batch['images'].shape}")
    print(f"Labels: {batch['labels']}")
    print(f"Modalities: {batch['modalities']}")
