import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import librosa
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
import json

class MultiModalDataset(Dataset):
    """
    Unified dataset for multi-modal sentiment analysis.
    Handles Text + Audio + Visual inputs.
    """
    def __init__(
        self, 
        data_dir: Path,
        split: str = 'train',
        max_audio_length: int = 16000 * 5  # 5 seconds at 16kHz
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_audio_length = max_audio_length
        self.logger = logging.getLogger(__name__)
        
        # Load metadata
        self.samples = self._load_metadata()
        
    def _load_metadata(self) -> List[Dict]:
        """
        Load dataset metadata. Scans data subdirectories if CSV is missing.
        """
        metadata_file = self.data_dir / f"{self.split}.csv"
        
        if metadata_file.exists():
            df = pd.read_csv(metadata_file)
            return df.to_dict('records')
        
        # Fallback: Scan directories for samples
        # This is a heuristic for when data is organized in subfolders
        self.logger.info(f"Scanning {self.data_dir} for samples...")
        samples = []
        
        # Look for text data
        text_dir = self.data_dir / "text_sentiment"
        audio_dir = self.data_dir / "audio_emotion"
        image_dir = self.data_dir / "image_emotion"
        
        # Simple alignment strategy: match by filename prefix/index
        if text_dir.exists():
            text_files = list(text_dir.glob("*.txt"))
            for i, tf in enumerate(text_files[:100]): # Limit for demo
                with open(tf, 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()
                
                # Try to find matching audio/image
                name = tf.stem
                audio_path = next(audio_dir.glob(f"{name}.*"), None)
                image_path = next(image_dir.glob(f"{name}.*"), None)
                
                samples.append({
                    'text': text_content,
                    'audio_path': str(audio_path) if audio_path else None,
                    'image_path': str(image_path) if image_path else None,
                    'label': 1 # Default neutral for unlabeled data
                })
        
        if not samples:
            return self._create_dummy_samples()
        
        return samples
    
    def _create_dummy_samples(self) -> List[Dict]:
        """Create dummy samples for testing when no real data exists"""
        dummy_samples = [
            {
                'text': 'I love this product! It is amazing!',
                'audio_path': None,
                'image_path': None,
                'label': 2  # Positive
            },
            {
                'text': 'This is terrible. I hate it.',
                'audio_path': None,
                'image_path': None,
                'label': 0  # Negative
            },
            {
                'text': 'It is okay, nothing special.',
                'audio_path': None,
                'image_path': None,
                'label': 1  # Neutral
            }
        ]
        return dummy_samples * 20  # Repeat for batch training
    
    def _load_audio(self, audio_path: Optional[str]) -> np.ndarray:
        """Load and preprocess audio file"""
        if audio_path is None or not Path(audio_path).exists():
            # Return dummy audio
            return np.random.randn(self.max_audio_length).astype(np.float32)
        
        # Load audio at 16kHz
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Pad or truncate
        if len(audio) > self.max_audio_length:
            audio = audio[:self.max_audio_length]
        else:
            audio = np.pad(audio, (0, self.max_audio_length - len(audio)))
        
        return audio.astype(np.float32)
    
    def _load_image(self, image_path: Optional[str]) -> Image.Image:
        """Load image"""
        if image_path is None or not Path(image_path).exists():
            # Return dummy image
            return Image.new('RGB', (224, 224), color='lightblue')
        
        image = Image.open(image_path).convert('RGB')
        return image
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        return {
            'text': sample['text'],
            'audio': self._load_audio(sample.get('audio_path')),
            'image': self._load_image(sample.get('image_path')),
            'label': sample['label']
        }

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching.
    Keeps text as list, converts audio/images appropriately.
    """
    texts = [item['text'] for item in batch]
    audio_data = [item['audio'] for item in batch]
    images = [item['image'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    return {
        'texts': texts,
        'audio_data': audio_data,
        'images': images,
        'labels': labels
    }
