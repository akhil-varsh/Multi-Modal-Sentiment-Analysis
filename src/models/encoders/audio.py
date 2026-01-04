from .base import BaseEncoder
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import numpy as np

class AudioEncoder(BaseEncoder):
    def __init__(self, model_name: str = 'facebook/wav2vec2-base-960h', frozen: bool = True):
        super().__init__(output_dim=768)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        
        if frozen:
            self.freeze()
            
    def forward(self, audio_data: list) -> torch.Tensor:
        """
        Args:
            audio_data: List of raw audio arrays (16kHz sampled)
        """
        device = next(self.parameters()).device
        
        # Process audio (Padding, Normalization)
        inputs = self.processor(
            audio_data, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        
        input_values = inputs.input_values.to(device)
        
        # Model Forward
        outputs = self.model(input_values)
        
        # Wav2Vec2 Output is Sequence (Batch, Seq_Len, 768)
        # We need a single global vector.
        # Strategy: Mean Pooling over time dimension
        features = outputs.last_hidden_state.mean(dim=1)
        
        return features
