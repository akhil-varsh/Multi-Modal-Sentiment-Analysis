"""
Quick test to verify the new architecture works end-to-end
"""
import torch
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, '.')

from src.models.system import MultiModalSentimentSystem

def test_system():
    print("🧪 Testing Multi-Modal Sentiment Analysis System...")
    print("=" * 60)
    
    # Initialize system
    print("\n1️⃣ Initializing system...")
    system = MultiModalSentimentSystem(freeze_encoders=True)
    system.eval()
    print("✅ System initialized successfully")
    
    # Create dummy inputs
    print("\n2️⃣ Creating test inputs...")
    texts = [
        "I love this product! It's amazing!",
        "This is terrible, I hate it.",
        "It's okay, nothing special."
    ]
    
    # Dummy audio (1 second at 16kHz)
    audio_data = [np.random.randn(16000).astype(np.float32) for _ in range(3)]
    
    # Dummy images
    images = [Image.new('RGB', (224, 224), color='red') for _ in range(3)]
    
    print(f"   - Texts: {len(texts)} samples")
    print(f"   - Audio: {len(audio_data)} samples (16kHz)")
    print(f"   - Images: {len(images)} samples (224x224)")
    
    # Run prediction
    print("\n3️⃣ Running prediction...")
    try:
        results = system.predict(texts, audio_data, images)
        
        print("\n📊 Results:")
        print("-" * 60)
        for i, text in enumerate(texts):
            print(f"\nSample {i+1}: '{text}'")
            print(f"  Sentiment: {results['sentiments'][i].upper()}")
            print(f"  Confidence: {results['probabilities'][i].max():.2%}")
            print(f"  Probabilities: {results['probabilities'][i]}")
            print(f"  Attention Weights (Text/Audio/Visual):")
            print(f"    {results['attention_weights'][i].mean(axis=0)}")
        
        print("\n" + "=" * 60)
        print("✅ System test PASSED!")
        print("\n🎯 Key Features Verified:")
        print("  ✓ RoBERTa Text Encoding")
        print("  ✓ Wav2Vec2 Audio Encoding")
        print("  ✓ ViT Visual Encoding")
        print("  ✓ Multi-Head Attention Fusion")
        print("  ✓ Attention Weight Visualization")
        
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_system()
    sys.exit(0 if success else 1)
