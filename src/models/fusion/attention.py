import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    """
    Multi-Head Attention-based Fusion for combining multi-modal features.
    This is the KEY component that differentiates this system from simple late fusion.
    """
    def __init__(
        self, 
        embed_dim: int = 768, 
        num_heads: int = 8, 
        dropout: float = 0.1,
        num_classes: int = 3
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-Head Self-Attention
        # This learns to weigh the importance of Text, Audio, and Visual modalities
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Normalization for stability
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # Feedforward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, text_features, audio_features, visual_features):
        """
        Args:
            text_features: (Batch, 768)
            audio_features: (Batch, 768)
            visual_features: (Batch, 768)
            
        Returns:
            logits: (Batch, num_classes)
            attention_weights: (Batch, 3, 3) - shows how each modality attends to others
        """
        # Stack features into sequence format
        # Shape: (Batch, 3, 768) where 3 = [Text, Audio, Visual]
        features = torch.stack([text_features, audio_features, visual_features], dim=1)
        
        # Multi-Head Self-Attention
        # This is where the magic happens - modalities "communicate" with each other
        attn_output, attention_weights = self.multihead_attn(
            features, features, features, 
            average_attn_weights=True
        )
        
        # Residual connection + Layer Norm
        features = self.layer_norm1(features + attn_output)
        
        # Feedforward Network
        ffn_output = self.ffn(features)
        features = self.layer_norm2(features + ffn_output)
        
        # Global pooling - average across modalities
        fused_features = features.mean(dim=1)  # (Batch, 768)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits, attention_weights
