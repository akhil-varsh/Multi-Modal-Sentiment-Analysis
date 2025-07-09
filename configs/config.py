"""
Configuration file for Multi-Modal Sentiment Analysis
"""

import torch
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" 
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    'csv_path': DATA_DIR / 'sample_multimodal' / 'sample_multimodal_data.csv',
    'features_dir': DATA_DIR / 'sample_multimodal' / 'features',
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
}

# Model configuration
MODEL_CONFIG = {
    'input_dim': 768,           # Feature dimension for all modalities
    'hidden_dim': 256,          # Hidden layer dimension
    'fusion_dim': 128,          # Fusion layer dimension
    'num_classes': 3,           # Number of sentiment classes (neg, neu, pos)
    'dropout_rate': 0.1,        # Dropout rate
    'attention_heads': 8,       # Number of attention heads in fusion
}

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 4,            # Batch size (small for sample data)
    'learning_rate': 1e-3,      # Learning rate
    'num_epochs': 25,           # Number of training epochs
    'weight_decay': 1e-5,       # L2 regularization
    'scheduler_step_size': 10,  # Learning rate scheduler step
    'scheduler_gamma': 0.8,     # Learning rate decay factor
    'early_stopping_patience': 7,  # Early stopping patience
    'gradient_clip_val': 1.0,   # Gradient clipping value
}

# Device configuration
DEVICE_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 2,           # Number of data loader workers
    'pin_memory': True,         # Pin memory for faster GPU transfer
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': LOGS_DIR / 'training.log',
}

# Evaluation configuration
EVAL_CONFIG = {
    'save_predictions': True,
    'save_attention_weights': True,
    'plot_confusion_matrix': True,
    'plot_training_curves': True,
    'analyze_attention_patterns': True,
}

# Model checkpoint configuration
CHECKPOINT_CONFIG = {
    'save_best_model': True,
    'save_last_model': True,
    'checkpoint_dir': MODELS_DIR,
    'best_model_name': 'best_multimodal_model.pth',
    'last_model_name': 'last_multimodal_model.pth',
}

# Inference configuration
INFERENCE_CONFIG = {
    'model_path': MODELS_DIR / 'best_multimodal_model.pth',
    'output_dir': RESULTS_DIR / 'predictions',
    'save_results': True,
}

# Visualization configuration
VIZ_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'save_plots': True,
    'plots_dir': RESULTS_DIR / 'plots',
    'plot_formats': ['png', 'pdf'],
}

# Create visualization directory
VIZ_CONFIG['plots_dir'].mkdir(exist_ok=True)

# Complete configuration dictionary
CONFIG = {
    'dataset': DATASET_CONFIG,
    'model': MODEL_CONFIG,
    'training': TRAINING_CONFIG,
    'device': DEVICE_CONFIG,
    'logging': LOGGING_CONFIG,
    'eval': EVAL_CONFIG,
    'checkpoint': CHECKPOINT_CONFIG,
    'inference': INFERENCE_CONFIG,
    'visualization': VIZ_CONFIG,
    'paths': {
        'project_root': PROJECT_ROOT,
        'data_dir': DATA_DIR,
        'models_dir': MODELS_DIR,
        'results_dir': RESULTS_DIR,
        'logs_dir': LOGS_DIR,
    }
}
