import torch
import os
import sys
from pathlib import Path


from src.data_loader import create_data_loaders, collate_fn
from src.feature_extractors import MultiModalSentimentSystem
from configs.config import CONFIG
from utils.helpers import setup_logging, set_seed, print_model_summary

def main():
    """Main training and evaluation pipeline"""
    
    # Setup logging and reproducibility
    setup_logging(CONFIG['logging']['file'], CONFIG['logging']['level'])
    set_seed(42)
    
    print("Multi-Modal Sentiment Analysis System")
    print("=" * 50)
    print(f"Device: {CONFIG['device']['device']}")
    print(f"Dataset: {CONFIG['dataset']['csv_path']}")
    
    # Create data loaders
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader = create_data_loaders(
        CONFIG['dataset']['csv_path'], 
        CONFIG['dataset']['features_dir'], 
        CONFIG['training']['batch_size']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Initialize model
    print("\nInitializing Simplified Multi-Modal Model...")
    model = MultiModalSentimentSystem()
    
    # Print model summary
    print_model_summary(model)
    
    # Test with sample prediction
    print("\nTesting sample prediction...")
    sample_texts = [
        "I love this product! It's amazing!",
        "This is terrible, I hate it."
    ]
    
    try:
        results = model.predict(sample_texts)
        
        print("Sample Predictions:")
        for i, text in enumerate(sample_texts):
            print(f"Text: '{text}'")
            print(f"Sentiment: {results['sentiments'][i]}")
            print(f"Confidence: {torch.max(results['probabilities'][i]).item():.3f}")
            print("-" * 40)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("This might be due to missing transformers dependency")
    
    print("\nBasic system test completed!")
    
    print(f"\n� Simplified Multimodal System Ready!")
    print("=" * 50)
    print("Text: RoBERTa-based sentiment analysis")
    print(" Audio: Simplified neural network") 
    print(" Visual: ViT (if available) or dummy features")
    print(" Fusion: Simple concatenation and neural network")
    print("=" * 50)
    print("\n Next steps:")
    print("1. Install transformers: pip install transformers")
    print("2. Add training loop for fine-tuning")
    print("3. Test with real audio/visual data")
    print(" System architecture is ready!")

if __name__ == "__main__":
    main()
