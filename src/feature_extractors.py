"""
Individual sentiment models for each modality
Using pre-trained models with fine-tuning approach
"""

import torch
import torch.nn as nn
import os
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    ViTImageProcessor, ViTModel
)
import numpy as np
from typing import List, Dict
import logging

# Check for HuggingFace token
HF_TOKEN = "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"
if HF_TOKEN:
    print(f"HuggingFace token found in environment")
    # Optionally login explicitly
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN)
        print("HuggingFace login successful")
    except Exception as e:
        print(f"HuggingFace login failed: {e}")
else:
    print("No HuggingFace token found in environment variables")

class TextSentimentModel(nn.Module):
    """
    Text sentiment analysis using fine-tuned RoBERTa
    Supports end-to-end fine-tuning
    """
    def __init__(self, num_labels=3, freeze_roberta=False):
        super().__init__()
        try:
            print("Loading RoBERTa model...")
            self.roberta = RobertaForSequenceClassification.from_pretrained(
                'roberta-base', 
                num_labels=num_labels,
                token=HF_TOKEN if HF_TOKEN else None  # Use 'token' instead of 'use_auth_token'
            )
            self.tokenizer = RobertaTokenizer.from_pretrained(
                'roberta-base',
                token=HF_TOKEN if HF_TOKEN else None  # Use 'token' instead of 'use_auth_token'
            )
            
            # Optionally freeze RoBERTa parameters  
            if freeze_roberta:
                for param in self.roberta.parameters():
                    param.requires_grad = False
            
            self.model_loaded = True
            self.num_labels = num_labels
            print("RoBERTa loaded successfully")
        except Exception as e:
            print(f"Failed to load RoBERTa: {e}")
            print("Falling back to dummy text model")
            self.model_loaded = False
            # Fallback dummy model
            self.dummy_classifier = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_labels)
            )
    
    def forward(self, input_ids=None, attention_mask=None, text_features=None):
        if self.model_loaded and input_ids is not None:
            # Real RoBERTa
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits
        else:
            # Dummy fallback
            if text_features is None:
                text_features = torch.randn(input_ids.size(0), 768)
            return self.dummy_classifier(text_features)
    
    def predict(self, texts: List[str]):
        """Predict sentiment for list of texts"""
        if self.model_loaded:
            encoded = self.tokenizer(
                texts, 
                truncation=True, 
                padding=True, 
                return_tensors='pt',
                max_length=512
            )
            
            with torch.no_grad():
                logits = self.forward(encoded['input_ids'], encoded['attention_mask'])
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
        else:
            # Dummy predictions
            batch_size = len(texts)
            logits = self.forward(text_features=torch.randn(batch_size, 768))
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        
        return predictions, probabilities


class AudioSentimentModel(nn.Module):
    """
    Audio sentiment analysis using Wav2Vec2 for feature extraction
    Supports end-to-end fine-tuning
    """
    def __init__(self, num_classes=3, freeze_wav2vec2=False, sr=16000):
        super().__init__()
        self.sr = sr
        try:
            print("Loading Wav2Vec2 model...")
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
            
            # Load pre-trained Wav2Vec2 model
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(
                'facebook/wav2vec2-base-960h',
                token=HF_TOKEN if HF_TOKEN else None
            )
            self.processor = Wav2Vec2Processor.from_pretrained(
                'facebook/wav2vec2-base-960h',
                token=HF_TOKEN if HF_TOKEN else None
            )
            
            # Optionally freeze Wav2Vec2 parameters
            if freeze_wav2vec2:
                for param in self.wav2vec2.parameters():
                    param.requires_grad = False
            
            # Classification head for sentiment
            self.classifier = nn.Sequential(
                nn.Linear(768, 256),  # Wav2Vec2 outputs 768-dim features
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)  # negative, neutral, positive
            )
            
            self.use_wav2vec2 = True
            self.num_classes = num_classes
            print("✅ Wav2Vec2 model loaded successfully")
            
        except Exception as e:
            print(f"⚠️ Wav2Vec2 not available: {e}")
            print("Using simplified audio processing")
            
            # Fallback: simplified audio processing
            self.classifier = nn.Sequential(
                nn.Linear(768, 256),  # Assume 768-dim input features
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
            self.use_wav2vec2 = False
            self.num_classes = num_classes
    
    def forward(self, audio_input):
        """
        Forward pass for audio sentiment classification.
        The entire preprocessing pipeline is encapsulated here to ensure
        correct gradient flow during end-to-end training.
        
        Args:
            audio_input: A list of raw audio waveforms (numpy arrays) or a pre-padded tensor.
        """
        if self.use_wav2vec2 and hasattr(self, 'wav2vec2'):
            device = next(self.wav2vec2.parameters()).device

            # The processor handles padding, normalization, and conversion to tensors.
            processed_inputs = self.processor(
                audio_input, 
                sampling_rate=self.sr, 
                return_tensors="pt", 
                padding=True
            ).to(device)

            # Extract features using Wav2Vec2
            wav2vec2_outputs = self.wav2vec2(processed_inputs.input_values)
            audio_features = wav2vec2_outputs.last_hidden_state.mean(dim=1)
            
            return self.classifier(audio_features)
        else:
            # Fallback: use pre-extracted features
            return self.classifier(audio_input)
    
    def extract_features_from_audio(self, audio_path: str) -> torch.Tensor:
        """
        Extracts features from a single audio file for inference.
        Note: For training, the forward pass should be used with raw audio data.
        """
        if not self.use_wav2vec2:
            # Fallback to MFCC features
            import librosa
            audio, _ = librosa.load(audio_path, sr=self.sr, duration=30)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
            
            # Combine features into a vector
            feature_vector = np.concatenate([
                np.mean(mfccs, axis=1),  # 13 MFCC features
                np.std(mfccs, axis=1),   # 13 MFCC std features
                [np.mean(spectral_centroids)],  # 1 spectral centroid
                [np.std(spectral_centroids)],   # 1 spectral centroid std
                [np.mean(zero_crossing_rate)],  # 1 ZCR
                [np.std(zero_crossing_rate)],   # 1 ZCR std
            ])
            
            # Pad or truncate to 768 dimensions
            if len(feature_vector) < 768:
                feature_vector = np.pad(feature_vector, (0, 768 - len(feature_vector)), 'constant')
            else:
                feature_vector = feature_vector[:768]
            
            return torch.FloatTensor(feature_vector)
        
        else:
            # Use Wav2Vec2 processor
            import librosa
            audio, _ = librosa.load(audio_path, sr=self.sr, duration=30)
            
            # Process audio for Wav2Vec2
            inputs = self.processor(
                audio, 
                sampling_rate=self.sr, 
                return_tensors="pt", 
                padding=True
            )
            
            # Extract features
            with torch.no_grad():
                outputs = self.wav2vec2(inputs.input_values)
                # Mean pooling over time dimension
                features = outputs.last_hidden_state.mean(dim=1).squeeze()
            
            return features


class VisualSentimentModel(nn.Module):
    """
    Visual sentiment analysis using ViT (with HuggingFace inference)
    Supports end-to-end fine-tuning
    """
    def __init__(self, num_classes=3, freeze_vit=False):
        super().__init__()
        try:
            print("Loading ViT model...")
            self.vit = ViTModel.from_pretrained(
                'google/vit-base-patch16-224',
                token=HF_TOKEN if HF_TOKEN else None
            )
            self.processor = ViTImageProcessor.from_pretrained(
                'google/vit-base-patch16-224',
                token=HF_TOKEN if HF_TOKEN else None
            )
            
            # Optionally freeze ViT parameters
            if freeze_vit:
                for param in self.vit.parameters():
                    param.requires_grad = False
            
            self.classifier = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
            self.use_vit = True
            self.num_classes = num_classes
            print("ViT model loaded successfully")
        except Exception as e:
            print(f"ViT not available: {e}")
            print("Using dummy visual features")
            self.use_vit = False
            self.num_classes = num_classes
            self.classifier = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, visual_input):
        """
        Forward pass for visual sentiment classification.
        The entire preprocessing pipeline is encapsulated here to ensure
        correct gradient flow during end-to-end training.
        
        Args:
            visual_input: A list of PIL Images.
        """
        if self.use_vit and hasattr(self, 'vit'):
            device = next(self.vit.parameters()).device

            # The processor handles conversion from PIL images to tensors,
            # including resizing and normalization.
            inputs = self.processor(images=visual_input, return_tensors="pt").to(device)
            pixel_values = inputs['pixel_values']
            
            # Extract features using ViT
            outputs = self.vit(pixel_values=pixel_values)
            pooled = outputs.last_hidden_state[:, 0]  # CLS token
            return self.classifier(pooled)
        else:
            # Fallback: use dummy features or handle tensor input
            if isinstance(visual_input, torch.Tensor) and visual_input.dim() == 2:
                # Already features
                return self.classifier(visual_input)
            else:
                # Create dummy features
                batch_size = len(visual_input)
                dummy_features = torch.randn(batch_size, 768)
                return self.classifier(dummy_features)


class MultiModalSentimentSystem(nn.Module):
    """
    Complete multimodal sentiment analysis system
    Combines text, audio, and visual models
    """
    def __init__(self):
        super().__init__()
        
        # Individual modality models
        self.text_model = TextSentimentModel()
        self.audio_model = AudioSentimentModel()
        self.visual_model = VisualSentimentModel()
        
        # Simple fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(9, 32),  # 3 models × 3 classes = 9 features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3)   # Final sentiment classes
        )
        
        logging.info("MultiModal Sentiment System initialized")
    
    def forward(self, texts=None, raw_audio=None, raw_images=None, text_inputs=None):
        # Handle both tokenized inputs and raw texts
        if text_inputs is not None:
            # Pre-tokenized inputs
            text_logits = self.text_model(text_inputs['input_ids'], text_inputs['attention_mask'])
        elif texts is not None and self.text_model.model_loaded:
            # Raw texts - tokenize them
            encoded = self.text_model.tokenizer(
                texts, truncation=True, padding=True, 
                return_tensors='pt', max_length=512
            )
            device = next(self.text_model.parameters()).device
            encoded = {k: v.to(device) for k, v in encoded.items()}
            text_logits = self.text_model(encoded['input_ids'], encoded['attention_mask'])
        else:
            # Fallback to dummy
            batch_size = len(texts) if texts else len(raw_audio)
            text_logits = self.text_model(text_features=torch.randn(batch_size, 768))
        
        # Get other modality predictions by passing raw data
        audio_logits = self.audio_model(raw_audio)
        visual_logits = self.visual_model(raw_images)
        
        # Concatenate logits for fusion
        combined = torch.cat([text_logits, audio_logits, visual_logits], dim=1)
        
        # Final fusion
        final_logits = self.fusion(combined)
        
        return {
            'final_logits': final_logits,
            'text_logits': text_logits,
            'audio_logits': audio_logits,
            'visual_logits': visual_logits
        }
    
    def predict(self, texts: List[str], audio_data: List = None, image_data: List = None):
        """End-to-end prediction with raw data"""
        batch_size = len(texts)
        
        # Use dummy data if not provided
        if audio_data is None:
            # Dummy raw audio waveform (1 second at 16kHz)
            audio_data = [np.random.randn(16000) for _ in range(batch_size)]
        if image_data is None:
            from PIL import Image
            # Dummy PIL image
            image_data = [Image.new('RGB', (224, 224), color='red') for _ in range(batch_size)]
        
        with torch.no_grad():
            outputs = self.forward(texts=texts, raw_audio=audio_data, raw_images=image_data)
            
            # Get final predictions
            probabilities = torch.softmax(outputs['final_logits'], dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
            
            # Convert to sentiment labels
            sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
            sentiments = [sentiment_map[pred.item()] for pred in predictions]
        
        return {
            'sentiments': sentiments,
            'probabilities': probabilities,
            'individual_outputs': {
                'text': torch.softmax(outputs['text_logits'], dim=-1),
                'audio': torch.softmax(outputs['audio_logits'], dim=-1),
                'visual': torch.softmax(outputs['visual_logits'], dim=-1)
            }
        }


if __name__ == "__main__":
    # Test the refactored system
    print("🧪 Testing Refactored Multimodal System...")
    
    # Sample data
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible, I hate it.",
        "It's okay, nothing special."
    ]
    
    # Initialize system
    try:
        system = MultiModalSentimentSystem()
        
        # The predict method now handles dummy data creation internally if not provided
        results = system.predict(test_texts)
        
        print("\n📊 Prediction Results (with dummy audio/visual):")
        for i, text in enumerate(test_texts):
            print(f"Text: '{text}'")
            print(f"  - Predicted Sentiment: {results['sentiments'][i]}")
            print(f"  - Confidence: {torch.max(results['probabilities'][i]).item():.3f}")
            print("-" * 40)
        
        print("✅ System test complete!")
        
    except Exception as e:
        import traceback
        print(f"❌ Error during testing: {e}")
        traceback.print_exc()
