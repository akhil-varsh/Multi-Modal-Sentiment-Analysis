from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import torch

@dataclass
class ModelConfig:
    # Modalitu Dimensions
    text_input_dim: int = 768
    audio_input_dim: int = 768
    visual_input_dim: int = 768
    
    # Advanced Settings
    hidden_dim: int = 256
    num_classes: int = 3
    dropout_rate: float = 0.1
    
    # Attention Fusion
    use_attention: bool = True
    attn_embed_dim: int = 768
    attn_num_heads: int = 8
    attn_dropout: float = 0.1

@dataclass
class TrainingConfig:
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 10
    weight_decay: float = 0.01
    warmup_steps: int = 500
    early_stopping_patience: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class ProjectConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Paths
    project_root: Path = Path(__file__).parent.parent.parent
    data_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    
    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

# Global Config Instance
config = ProjectConfig()
