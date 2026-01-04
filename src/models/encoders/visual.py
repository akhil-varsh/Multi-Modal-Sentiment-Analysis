from .base import BaseEncoder
import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor
from PIL import Image

class VisualEncoder(BaseEncoder):
    def __init__(self, model_name: str = 'google/vit-base-patch16-224', frozen: bool = True):
        super().__init__(output_dim=768)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        
        if frozen:
            self.freeze()
            
    def forward(self, images: list[Image.Image]) -> torch.Tensor:
        """
        Args:
            images: List of PIL Images
        """
        device = next(self.parameters()).device
        
        # Process images
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(device)
        
        # Model Forward
        outputs = self.model(pixel_values=pixel_values)
        
        # ViT Output has a CLS token at index 0
        # Shape: (Batch, 768)
        features = outputs.last_hidden_state[:, 0, :]
        return features
