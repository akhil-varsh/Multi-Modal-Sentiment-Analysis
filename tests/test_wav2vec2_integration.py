#!/usr/bin/env python3
"""
Test script for Wav2Vec2 integration in audio sentiment analysis
"""

import sys
from pathlib import Path
import torch
import os
import tempfile
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from feature_extractors import AudioSentimentModel

def create_dummy_audio_file(duration=2, sample_rate=16000):
    """Create a dummy audio file for testing"""
    # Generate a simple sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        # Write a simple WAV file (this is a simplified version)
        import wave
        with wave.open(tmp_file.name, 'w') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
        
        return tmp_file.name

def test_wav2vec2_integration():
    """Test Wav2Vec2 integration in AudioSentimentModel"""
    print("🔍 Testing Wav2Vec2 Integration...")
    
    # Initialize audio sentiment model
    print("📝 Initializing AudioSentimentModel...")
    audio_model = AudioSentimentModel()
    
    # Check if Wav2Vec2 is available
    if hasattr(audio_model, 'use_wav2vec2') and audio_model.use_wav2vec2:
        print("✅ Wav2Vec2 is available!")
        
        # Create dummy audio file
        print("🎵 Creating dummy audio file...")
        audio_file = create_dummy_audio_file()
        
        try:
            # Test feature extraction
            print("🔧 Testing feature extraction...")
            features = audio_model.extract_features_from_audio(audio_file)
            print(f"✅ Feature extraction successful! Shape: {features.shape}")
            
            # Test forward pass
            print("🚀 Testing forward pass...")
            features_batch = features.unsqueeze(0)  # Add batch dimension
            output = audio_model(features_batch)
            print(f"✅ Forward pass successful! Output shape: {output.shape}")
            
            # Test sentiment prediction
            print("💭 Testing sentiment prediction...")
            probabilities = torch.softmax(output, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)
            print(f"✅ Sentiment prediction: Class {predicted_class.item()}")
            print(f"   Probabilities: {probabilities.squeeze().tolist()}")
            
            # Clean up
            os.unlink(audio_file)
            print("🧹 Cleaned up temporary file")
            
        except Exception as e:
            print(f"❌ Error during testing: {e}")
            # Clean up on error
            if os.path.exists(audio_file):
                os.unlink(audio_file)
            return False
            
    else:
        print("⚠️ Wav2Vec2 is not available, using fallback MFCC features")
        
        # Create dummy audio file
        print("🎵 Creating dummy audio file...")
        audio_file = create_dummy_audio_file()
        
        try:
            # Test fallback feature extraction
            print("🔧 Testing fallback feature extraction...")
            features = audio_model.extract_features_from_audio(audio_file)
            print(f"✅ Feature extraction successful! Shape: {features.shape}")
            
            # Test forward pass
            print("🚀 Testing forward pass...")
            features_batch = features.unsqueeze(0)  # Add batch dimension
            output = audio_model(features_batch)
            print(f"✅ Forward pass successful! Output shape: {output.shape}")
            
            # Test sentiment prediction
            print("💭 Testing sentiment prediction...")
            probabilities = torch.softmax(output, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)
            print(f"✅ Sentiment prediction: Class {predicted_class.item()}")
            print(f"   Probabilities: {probabilities.squeeze().tolist()}")
            
            # Clean up
            os.unlink(audio_file)
            print("🧹 Cleaned up temporary file")
            
        except Exception as e:
            print(f"❌ Error during testing: {e}")
            # Clean up on error
            if os.path.exists(audio_file):
                os.unlink(audio_file)
            return False
    
    print("🎉 All tests passed!")
    return True

def test_real_audio_file():
    """Test with a real audio file if available"""
    print("\n🔍 Testing with real audio files...")
    
    # Look for sample audio files in the data directory
    audio_dir = Path(__file__).parent.parent / 'data' / 'audio_emotion' / 'emodb' / 'wav'
    
    if audio_dir.exists():
        audio_files = list(audio_dir.glob('*.wav'))[:3]  # Test with first 3 files
        
        if audio_files:
            print(f"📁 Found {len(audio_files)} audio files to test")
            
            # Initialize audio model
            audio_model = AudioSentimentModel()
            
            for audio_file in audio_files:
                print(f"\n🎵 Testing with: {audio_file.name}")
                
                try:
                    # Extract features
                    features = audio_model.extract_features_from_audio(str(audio_file))
                    print(f"   ✅ Feature shape: {features.shape}")
                    
                    # Get prediction
                    features_batch = features.unsqueeze(0)
                    output = audio_model(features_batch)
                    probabilities = torch.softmax(output, dim=-1)
                    predicted_class = torch.argmax(probabilities, dim=-1)
                    
                    print(f"   🎯 Predicted sentiment: Class {predicted_class.item()}")
                    print(f"   📊 Probabilities: {[f'{p:.3f}' for p in probabilities.squeeze().tolist()]}")
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
        else:
            print("⚠️ No audio files found in data directory")
    else:
        print("⚠️ Audio data directory not found")

if __name__ == "__main__":
    print("🧪 Wav2Vec2 Integration Test Suite")
    print("=" * 50)
    
    # Test 1: Basic integration
    success = test_wav2vec2_integration()
    
    if success:
        # Test 2: Real audio files
        test_real_audio_file()
    
    print("\n" + "=" * 50)
    print("🏁 Test suite completed!")
