from abc import ABC, abstractmethod
import torch.nn as nn
import torch

class BaseEncoder(nn.Module, ABC):
    """
    Abstract Base Class for all modality encoders.
    Enforces a common interface for feature extraction.
    """
    def __init__(self, output_dim: int = 768):
        super().__init__()
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x) -> torch.Tensor:
        """
        Forward pass to extract features.
        
        Args:
            x: Input data for the modality.
            
        Returns:
            torch.Tensor: Feature embeddings of shape (Batch_Size, Output_Dim)
        """
        pass
    
    def freeze(self):
        """Freeze all parameters in the encoder."""
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        """Unfreeze all parameters in the encoder."""
        for param in self.parameters():
            param.requires_grad = True
