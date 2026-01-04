import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from PIL import Image
import numpy as np

from .encoders.text import TextEncoder
from .encoders.audio import AudioEncoder
from .encoders.visual import VisualEncoder
from .fusion.attention import AttentionFusion

class MultiModalSentimentSystem(nn.Module):
    """
    Complete Multi-Modal Sentiment Analysis System with Attention-Based Fusion.
    This is the main interface for the entire system.
    """
    def __init__(
        self, 
        freeze_encoders: bool = True,
        num_classes: int = 3
    ):
        super().__init__()
        
        # Individual Encoders
        self.text_encoder = TextEncoder(frozen=freeze_encoders)
        self.audio_encoder = AudioEncoder(frozen=freeze_encoders)
        self.visual_encoder = VisualEncoder(frozen=freeze_encoders)
        
        # Attention Fusion
        self.fusion = AttentionFusion(
            embed_dim=768,
            num_heads=8,
            dropout=0.1,
            num_classes=num_classes
        )
        
        self.sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        
    def forward(
        self, 
        texts: List[str], 
        audio_data: List[np.ndarray], 
        images: List[Image.Image]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the entire system.
        
        Args:
            texts: List of text strings
            audio_data: List of raw audio waveforms (16kHz)
            images: List of PIL Images
            
        Returns:
            logits: (Batch, num_classes)
            attention_weights: (Batch, 3, 3)
        """
        # Extract features from each modality
        text_features = self.text_encoder(texts)
        audio_features = self.audio_encoder(audio_data)
        visual_features = self.visual_encoder(images)
        
        # Fusion with Attention
        logits, attention_weights = self.fusion(
            text_features, audio_features, visual_features
        )
        
        return logits, attention_weights
    
    @torch.no_grad()
    def predict(
        self, 
        texts: List[str], 
        audio_data: List[np.ndarray], 
        images: List[Image.Image]
    ) -> Dict:
        """
        End-to-end prediction with interpretability.
        
        Returns:
            Dictionary containing predictions, probabilities, and attention weights
        """
        self.eval()
        
        logits, attention_weights = self.forward(texts, audio_data, images)
        
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        
        # Convert to sentiment labels
        sentiments = [self.sentiment_map[pred.item()] for pred in predictions]
        
        return {
            'sentiments': sentiments,
            'probabilities': probabilities.cpu().numpy(),
            'attention_weights': attention_weights.cpu().numpy(),
            'logits': logits.cpu().numpy()
        }
