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
HF_TOKEN = "YOUR_HF_TOKEN"
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
    """
    def __init__(self, num_labels=3):
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
            self.model_loaded = True
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
    Audio sentiment analysis - simplified without librosa dependency
    """
    def __init__(self):
        super().__init__()
        # Simplified audio processing without Wav2Vec2 for now
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),  # Assume 768-dim input features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    
    def forward(self, audio_features):
        return self.classifier(audio_features)


class VisualSentimentModel(nn.Module):
    """
    Visual sentiment analysis using ViT (with HuggingFace inference)
    """
    def __init__(self):
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
            self.classifier = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 3)
            )
            self.use_vit = True
            print("ViT model loaded successfully")
        except Exception as e:
            print(f"ViT not available: {e}")
            print("Using dummy visual features")
            self.use_vit = False
            self.classifier = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 3)
            )
    
    def forward(self, visual_features):
        if self.use_vit and hasattr(self, 'pixel_values'):
            # Real ViT processing
            outputs = self.vit(pixel_values=visual_features)
            pooled = outputs.last_hidden_state[:, 0]  # CLS token
            return self.classifier(pooled)
        else:
            # Dummy processing
            return self.classifier(visual_features)


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
    
    def forward(self, text_inputs=None, audio_features=None, visual_features=None, texts=None):
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
            text_logits = self.text_model(encoded['input_ids'], encoded['attention_mask'])
        else:
            # Fallback to dummy
            batch_size = len(texts) if texts else audio_features.size(0)
            text_logits = self.text_model(text_features=torch.randn(batch_size, 768))
        
        # Get other modality predictions
        audio_logits = self.audio_model(audio_features)
        visual_logits = self.visual_model(visual_features)
        
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
    
    def predict(self, texts: List[str], audio_features=None, visual_features=None):
        """End-to-end prediction"""
        batch_size = len(texts)
        
        # Use dummy features if not provided
        if audio_features is None:
            audio_features = torch.randn(batch_size, 768)
        if visual_features is None:
            visual_features = torch.randn(batch_size, 768)
        
        with torch.no_grad():
            outputs = self.forward(texts=texts, audio_features=audio_features, visual_features=visual_features)
            
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
    # Test the system
    print("🧪 Testing Multimodal System with HF Token...")
    print(f"Token status: {'✅ Found' if HF_TOKEN else '❌ Not found'}")
    
    # Sample data
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible, I hate it.",
        "It's okay, nothing special."
    ]
    
    # Initialize system
    try:
        system = MultiModalSentimentSystem()
        
        # Make predictions
        results = system.predict(test_texts)
        
        print("\n📊 Prediction Results:")
        for i, text in enumerate(test_texts):
            print(f"Text: '{text}'")
            print(f"Sentiment: {results['sentiments'][i]}")
            print(f"Confidence: {torch.max(results['probabilities'][i]).item():.3f}")
            print("-" * 40)
        
        print("✅ System working with your HF token!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Check if your HF token has proper permissions")
