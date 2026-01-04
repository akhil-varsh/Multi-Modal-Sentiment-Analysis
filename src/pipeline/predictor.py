import torch
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from PIL import Image

from ..models.system import MultiModalSentimentSystem

class Predictor:
    """
    Inference pipeline for trained Multi-Modal Sentiment Analysis models
    """
    def __init__(
        self, 
        checkpoint_path: Optional[Path] = None,
        device: str = 'cpu'
    ):
        self.device = device
        
        # Initialize model
        self.model = MultiModalSentimentSystem(num_classes=3)
        
        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)
            print(f"✅ Loaded model from {checkpoint_path}")
        else:
            print("⚠️ No checkpoint loaded - using untrained model")
        
        self.model.to(device)
        self.model.eval()
        
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model weights from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
    @torch.no_grad()
    def predict(
        self,
        texts: List[str],
        audio_data: List[np.ndarray],
        images: List[Image.Image]
    ) -> Dict:
        """
        Run inference on input data
        
        Args:
            texts: List of text strings
            audio_data: List of audio waveforms (16kHz)
            images: List of PIL Images
            
        Returns:
            Dictionary with predictions, probabilities, and attention weights
        """
        return self.model.predict(texts, audio_data, images)
    
    def predict_single(
        self,
        text: str,
        audio: np.ndarray,
        image: Image.Image
    ) -> Dict:
        """
        Convenience method for single sample prediction
        """
        result = self.predict([text], [audio], [image])
        
        return {
            'sentiment': result['sentiments'][0],
            'probabilities': result['probabilities'][0],
            'attention_weights': result['attention_weights'][0]
        }
