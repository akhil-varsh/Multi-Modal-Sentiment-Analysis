#!/usr/bin/env python3
"""
Test script for end-to-end multi-modal training
Verifies that all components work together for proper fine-tuning
"""

import sys
from pathlib import Path
import torch
import os
import tempfile
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent / 'training'))

def test_end_to_end_model():
    """Test the end-to-end model initialization and forward pass"""
    print("🧪 Testing End-to-End Model")
    print("-" * 40)
    
    try:
        from train_end_to_end import EndToEndMultiModalModel
        
        # Initialize model
        print("📝 Initializing EndToEndMultiModalModel...")
        model = EndToEndMultiModalModel(num_classes=3)
        
        print("✅ Model initialized successfully")
        
        # Test model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")
        
        # Check individual components
        print("\n🔍 Component Check:")
        print(f"   Text model loaded: {model.text_model.model_loaded if hasattr(model.text_model, 'model_loaded') else 'Unknown'}")
        print(f"   Audio Wav2Vec2: {model.audio_model.use_wav2vec2 if hasattr(model.audio_model, 'use_wav2vec2') else 'Unknown'}")
        print(f"   Visual ViT: {model.visual_model.use_vit if hasattr(model.visual_model, 'use_vit') else 'Unknown'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_individual_components():
    """Test individual model components"""
    print("\n🧪 Testing Individual Components")
    print("-" * 40)
    
    try:
        from feature_extractors import TextSentimentModel, AudioSentimentModel, VisualSentimentModel
        
        # Test Text Model
        print("📝 Testing TextSentimentModel...")
        text_model = TextSentimentModel(num_labels=3, freeze_roberta=False)
        print(f"   ✅ Text model: loaded={text_model.model_loaded}, classes={text_model.num_labels}")
        
        # Test Audio Model  
        print("📝 Testing AudioSentimentModel...")
        audio_model = AudioSentimentModel(num_classes=3, freeze_wav2vec2=False)
        print(f"   ✅ Audio model: wav2vec2={audio_model.use_wav2vec2}, classes={audio_model.num_classes}")
        
        # Test Visual Model
        print("📝 Testing VisualSentimentModel...")  
        visual_model = VisualSentimentModel(num_classes=3, freeze_vit=False)
        print(f"   ✅ Visual model: vit={visual_model.use_vit}, classes={visual_model.num_classes}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_data_loading():
    """Test data loading for end-to-end training"""
    print("\n🧪 Testing Data Loading")
    print("-" * 40)
    
    try:
        from real_data_loader import get_data_config, create_real_data_loaders
        
        print("📁 Loading data configuration...")
        config = get_data_config()
        print("✅ Configuration loaded")
        
        print("📂 Creating data loaders...")
        train_loader, val_loader, test_loader = create_real_data_loaders(
            config, 
            batch_size=2  # Small batch for testing
        )
        
        print(f"✅ Data loaders created:")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")  
        print(f"   Test batches: {len(test_loader)}")
        
        # Test loading one batch
        print("\n🔄 Testing batch loading...")
        for batch in train_loader:
            print(f"   ✅ Batch loaded with keys: {list(batch.keys())}")
            if 'label' in batch:
                print(f"   📊 Batch size: {len(batch['label'])}")
            break
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_forward_pass():
    """Test forward pass with dummy data"""
    print("\n🧪 Testing Forward Pass")
    print("-" * 40)
    
    try:
        from train_end_to_end import EndToEndMultiModalModel
        
        # Initialize model
        model = EndToEndMultiModalModel(num_classes=3)
        model.eval()
        
        # Create dummy inputs
        batch_size = 2
        
        # Dummy text inputs (if RoBERTa is available)
        text_inputs = None
        text_attention_mask = None
        if hasattr(model.text_model, 'tokenizer'):
            dummy_texts = ["This is a positive sentence.", "This is a negative sentence."]
            encoded = model.text_model.tokenizer(
                dummy_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            text_inputs = encoded['input_ids']
            text_attention_mask = encoded['attention_mask']
            print(f"   📝 Text inputs shape: {text_inputs.shape}")
        
        # Dummy audio inputs (create temporary audio files)
        audio_inputs = None
        if model.audio_model.use_wav2vec2:
            print("   🎵 Creating dummy audio files...")
            audio_files = []
            for i in range(batch_size):
                # Create a temporary WAV file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    import wave
                    sample_rate = 16000
                    duration = 1
                    t = np.linspace(0, duration, int(sample_rate * duration), False)
                    audio_data = np.sin(2 * np.pi * 440 * t)  # Simple sine wave
                    
                    with wave.open(tmp_file.name, 'w') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
                    
                    audio_files.append(tmp_file.name)
            
            audio_inputs = audio_files
            print(f"   🎵 Created {len(audio_files)} audio files")
        
        # Dummy visual inputs  
        visual_inputs = None
        if model.visual_model.use_vit:
            from PIL import Image
            # Create dummy RGB images
            visual_inputs = []
            for i in range(batch_size):
                # Create a random RGB image
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                visual_inputs.append(img)
            print(f"   🖼️ Created {len(visual_inputs)} dummy images")
        
        # Test forward pass
        print("\n🚀 Running forward pass...")
        with torch.no_grad():
            outputs = model(
                text_inputs=text_inputs,
                audio_inputs=audio_inputs,
                visual_inputs=visual_inputs,
                text_attention_mask=text_attention_mask
            )
        
        print(f"✅ Forward pass successful!")
        print(f"   Output shape: {outputs.shape}")
        print(f"   Output sample: {outputs[0].tolist()}")
        
        # Test with attention weights
        print("\n🔍 Testing attention mechanism...")
        with torch.no_grad():
            outputs, attention = model(
                text_inputs=text_inputs,
                audio_inputs=audio_inputs, 
                visual_inputs=visual_inputs,
                text_attention_mask=text_attention_mask,
                return_attention=True
            )
        
        print(f"✅ Attention mechanism working!")
        print(f"   Attention shape: {attention.shape}")
        
        # Clean up audio files
        if audio_inputs:
            for audio_file in audio_inputs:
                try:
                    os.unlink(audio_file)
                except:
                    pass
            print("🧹 Cleaned up temporary audio files")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_training_setup():
    """Test training setup and optimizer configuration"""
    print("\n🧪 Testing Training Setup")
    print("-" * 40)
    
    try:
        from train_end_to_end import EndToEndMultiModalModel, EndToEndTrainer
        
        # Initialize model and trainer
        model = EndToEndMultiModalModel(num_classes=3)
        trainer = EndToEndTrainer(
            model=model,
            device='cpu',
            learning_rate=2e-5,
            warmup_steps=100
        )
        
        print("✅ Trainer initialized successfully")
        
        # Check parameter groups
        print(f"📊 Optimizer parameter groups: {len(trainer.optimizer.param_groups)}")
        for i, group in enumerate(trainer.optimizer.param_groups):
            print(f"   Group {i}: {len(group['params'])} params, lr={group['lr']}")
        
        # Test scheduler setup
        trainer.setup_scheduler(total_steps=1000)
        print("✅ Scheduler setup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Run all end-to-end tests"""
    print("🚀 End-to-End Multi-Modal Training Test Suite")
    print("=" * 60)
    
    # Check if running from correct directory
    if not Path('src/feature_extractors.py').exists():
        print("❌ Please run this script from the project root directory:")
        print("   cd 'c:\\Users\\Akhil\\Python_Projects\\ML\\Multi Modal-Sentiment Analysis'")
        print("   python tests/test_end_to_end_training.py")
        return
    
    tests = [
        test_individual_components,
        test_end_to_end_model,
        test_data_loading,
        test_forward_pass,
        test_training_setup
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Unexpected error in {test.__name__}: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    
    test_names = [
        "Individual Components",
        "End-to-End Model",
        "Data Loading",
        "Forward Pass",
        "Training Setup"
    ]
    
    passed = 0
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {i+1}. {name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! End-to-end training is ready!")
        print("\n🚀 You can now run:")
        print("   python training/train_end_to_end.py --epochs 5 --batch_size 4")
        print("\n💡 Recommended training command:")
        print("   python training/train_end_to_end.py \\")
        print("       --epochs 10 \\")
        print("       --batch_size 8 \\") 
        print("       --learning_rate 2e-5 \\")
        print("       --warmup_steps 1000 \\")
        print("       --device cpu")
    else:
        print(f"\n⚠️ {len(tests) - passed} test(s) failed. Please check the errors above.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
