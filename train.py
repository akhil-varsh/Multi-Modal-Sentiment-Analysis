"""
Main training script for Multi-Modal Sentiment Analysis
Usage: python train.py
"""
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.models.system import MultiModalSentimentSystem
from src.data.dataset import MultiModalDataset, collate_fn
from src.training.trainer import Trainer

def main():
    print("=" * 60)
    print("Multi-Modal Sentiment Analysis - Training")
    print("=" * 60)
    
    # Configuration
    batch_size = 4
    num_epochs = 10
    learning_rate = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n📋 Configuration:")
    print(f"  - Device: {device}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Learning Rate: {learning_rate}")
    
    # Create datasets
    print("\n📂 Loading datasets...")
    data_dir = Path('data')
    
    train_dataset = MultiModalDataset(data_dir, split='train')
    val_dataset = MultiModalDataset(data_dir, split='val')
    
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Initialize model
    print("\n🤖 Initializing model...")
    model = MultiModalSentimentSystem(
        freeze_encoders=True,  # Keep pretrained weights frozen initially
        num_classes=3
    )
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        checkpoint_dir=Path('models')
    )
    
    # Train
    print("\n🚀 Starting training...\n")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
