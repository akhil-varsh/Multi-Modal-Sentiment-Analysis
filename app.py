#!/usr/bin/env python3
"""
Enhanced Multi-Modal Sentiment Analysis Prediction System
Uses the trained real-data model for inference on new inputs
"""


import torch
import torch.nn as nn
import argparse
import sys
import os
from pathlib import Path
import logging
from PIL import Image
import numpy as np
import librosa
from typing import Dict, Any, Optional, Union

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.feature_extractors import TextSentimentModel, AudioSentimentModel, VisualSentimentModel
from src.models.fusion_model import MultiModalSentimentAnalyzer
from src.real_data_loader import get_data_config

class EnhancedMultiModalPredictor:
    """Enhanced predictor that can handle raw inputs and trained models"""
    
    def __init__(self, 
                 fusion_model_path: str = '../models/multimodal_sentiment_real_best.pth',
                 device: str = 'cpu'):
        """
        Initialize the enhanced predictor
        
        Args:
            fusion_model_path: Path to the trained fusion model
            device: Device to run inference on
        """
        self.device = torch.device(device)
        self.setup_logging()
        
        # Initialize feature extractors
        logging.info("Loading feature extraction models...")
        self.feature_extractors = {
            'text': TextSentimentModel(num_labels=3),
            'audio': AudioSentimentModel(),
            'visual': VisualSentimentModel()
        }
        
        # Move to device
        for name, model in self.feature_extractors.items():
            model.to(self.device)
            model.eval()
            logging.info(f"✓ {name.title()} model loaded")
        
        # Initialize fusion model
        self.fusion_model = MultiModalSentimentAnalyzer(
            input_dim=2,  # Binary features
            hidden_dim=256,
            fusion_dim=128,
            num_classes=2  # Binary sentiment
        )
        
        # Load trained weights if available
        if os.path.exists(fusion_model_path):
            try:
                self.fusion_model.load_state_dict(torch.load(fusion_model_path, map_location=self.device))
                logging.info(f"✓ Trained fusion model loaded from {fusion_model_path}")
            except Exception as e:
                logging.warning(f"Failed to load fusion model: {e}")
                logging.info("Using untrained fusion model")
        else:
            logging.warning(f"Fusion model not found at {fusion_model_path}")
            logging.info("Using untrained fusion model")
        
        self.fusion_model.to(self.device)
        self.fusion_model.eval()
        
        # Sentiment mappings
        self.sentiment_labels = {0: "Negative", 1: "Positive"}
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def process_text(self, text: str) -> torch.Tensor:
        """Process text input and extract features"""
        if not text or not text.strip():
            # Return dummy features for empty text
            return torch.zeros(1, 3).to(self.device)
        
        try:
            # Use the text model to get probabilities
            predictions, probabilities = self.feature_extractors['text'].predict([text])
            return probabilities.to(self.device)
        except Exception as e:
            logging.warning(f"Text processing failed: {e}")
            return torch.zeros(1, 3).to(self.device)
    
    def process_audio(self, audio_path: str) -> torch.Tensor:
        """Process audio file and extract features using Wav2Vec2"""
        if not audio_path or not os.path.exists(audio_path):
            # Return dummy features for missing audio
            return torch.zeros(1, 3).to(self.device)
        
        try:
            # Check if audio model supports Wav2Vec2
            if hasattr(self.feature_extractors['audio'], 'use_wav2vec2') and self.feature_extractors['audio'].use_wav2vec2:
                # Use Wav2Vec2 feature extraction
                audio_features = self.feature_extractors['audio'].extract_features_from_audio(audio_path)
                audio_features = audio_features.unsqueeze(0).to(self.device)  # Add batch dimension
                
                # Get predictions
                with torch.no_grad():
                    features = self.feature_extractors['audio'](audio_features)
                    probabilities = torch.softmax(features, dim=-1)
            else:
                # Fallback to MFCC features
                import librosa
                audio, sr = librosa.load(audio_path, sr=16000, duration=30)
                
                # Extract basic audio features (MFCCs, spectral features, etc.)
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
                zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
                
                # Combine features into a 768-dimensional vector
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
                
                # Convert to tensor and add batch dimension
                audio_features = torch.FloatTensor(feature_vector).unsqueeze(0).to(self.device)
                
                # Extract features using the actual audio model
                with torch.no_grad():
                    features = self.feature_extractors['audio'](audio_features)
                    probabilities = torch.softmax(features, dim=-1)
            
            return probabilities
        except Exception as e:
            logging.warning(f"Audio processing failed: {e}")
            return torch.zeros(1, 3).to(self.device)
    
    def process_image(self, image_path: str) -> torch.Tensor:
        """Process image file and extract features"""
        if not image_path or not os.path.exists(image_path):
            # Return dummy features for missing image
            return torch.zeros(1, 3).to(self.device)
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Get features from visual model
            with torch.no_grad():
                features = self.feature_extractors['visual'](image)
                probabilities = torch.softmax(features, dim=-1)
            
            return probabilities
        except Exception as e:
            logging.warning(f"Image processing failed: {e}")
            return torch.zeros(1, 3).to(self.device)
    
    def convert_3class_to_binary(self, probs_3class: torch.Tensor) -> torch.Tensor:
        """Convert 3-class probabilities to binary sentiment features"""
        # Assume: class 0=negative, class 1=neutral, class 2=positive
        # Binary: 0=negative, 1=positive (combine neutral+positive)
        negative_prob = probs_3class[:, 0:1]  # Keep negative as is
        positive_prob = probs_3class[:, 1:].sum(dim=1, keepdim=True)  # Neutral + Positive
        return torch.cat([negative_prob, positive_prob], dim=1)
    
    def predict_sentiment(self, 
                         text: str = None,
                         audio_path: str = None,
                         image_path: str = None) -> Dict[str, Any]:
        """
        Predict sentiment from multimodal inputs
        
        Args:
            text: Text input string
            audio_path: Path to audio file
            image_path: Path to image file
            
        Returns:
            Dictionary with prediction results
        """
        
        logging.info("Starting multimodal sentiment prediction...")
        
        # Process each modality
        text_features = self.process_text(text) if text else torch.zeros(1, 3).to(self.device)
        audio_features = self.process_audio(audio_path) if audio_path else torch.zeros(1, 3).to(self.device)
        visual_features = self.process_image(image_path) if image_path else torch.zeros(1, 3).to(self.device)
        
        # Convert to binary features for fusion model
        text_binary = self.convert_3class_to_binary(text_features)
        audio_binary = self.convert_3class_to_binary(audio_features)
        visual_binary = self.convert_3class_to_binary(visual_features)
        
        # Predict using fusion model
        with torch.no_grad():
            outputs, attention_weights = self.fusion_model(
                text_binary, audio_binary, visual_binary
            )
            
            probabilities = torch.softmax(outputs, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)
        
        # Format results
        result = {
            'sentiment': self.sentiment_labels[predicted_class.item()],
            'confidence': torch.max(probabilities).item(),
            'probabilities': {
                'negative': probabilities[0][0].item(),
                'positive': probabilities[0][1].item()
            },
            'attention_weights': {
                'text': attention_weights['text_attention'][0].item(),
                'audio': attention_weights['audio_attention'][0].item(),
                'visual': attention_weights['visual_attention'][0].item()
            },
            'individual_predictions': {
                'text': {
                    'negative': text_features[0][0].item(),
                    'neutral': text_features[0][1].item(),
                    'positive': text_features[0][2].item()
                },
                'audio': {
                    'negative': audio_features[0][0].item(),
                    'neutral': audio_features[0][1].item(),
                    'positive': audio_features[0][2].item()
                },
                'visual': {
                    'negative': visual_features[0][0].item(),
                    'neutral': visual_features[0][1].item(),
                    'positive': visual_features[0][2].item()
                }
            }
        }
        
        return result
    
    def print_results(self, result: Dict[str, Any], 
                     text: str = None, 
                     audio_path: str = None, 
                     image_path: str = None):
        """Print formatted prediction results"""
        
        print("\n" + "="*60)
        print("🔮 MULTIMODAL SENTIMENT ANALYSIS RESULTS")
        print("="*60)
        
        # Input summary
        print("📝 Inputs:")
        if text:
            print(f"   Text: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
        if audio_path:
            print(f"   Audio: {audio_path}")
        if image_path:
            print(f"   Image: {image_path}")
        
        print()
        
        # Main result
        print("🎯 Overall Prediction:")
        print(f"   Sentiment: {result['sentiment']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print()
        
        # Probability breakdown
        print("📊 Probability Distribution:")
        print(f"   Negative: {result['probabilities']['negative']:.1%}")
        print(f"   Positive: {result['probabilities']['positive']:.1%}")
        print()
        
        # Attention weights
        print("🎛️ Attention Weights:")
        print(f"   Text:   {result['attention_weights']['text']:.1%}")
        print(f"   Audio:  {result['attention_weights']['audio']:.1%}")
        print(f"   Visual: {result['attention_weights']['visual']:.1%}")
        print()
        
        # Individual modality predictions
        print("🔍 Individual Modality Predictions:")
        for modality in ['text', 'audio', 'visual']:
            preds = result['individual_predictions'][modality]
            print(f"   {modality.title()}:")
            print(f"     Negative: {preds['negative']:.1%}")
            print(f"     Neutral:  {preds['neutral']:.1%}")
            print(f"     Positive: {preds['positive']:.1%}")
        
        print("="*60)

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Enhanced Multi-Modal Sentiment Analysis')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--audio', type=str, help='Path to audio file')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--model', type=str, default='multimodal_sentiment_real_best.pth',
                       help='Path to trained fusion model')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use for inference')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        predictor = EnhancedMultiModalPredictor(
            fusion_model_path=args.model,
            device=args.device
        )
        print("✅ Enhanced predictor initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        return
    
    if args.interactive:
        # Interactive mode
        print("\n🎯 Interactive Multi-Modal Sentiment Analysis")
        print("Enter inputs for analysis (press Enter to skip a modality)")
        print("Type 'quit' to exit\n")
        
        while True:
            print("-" * 40)
            text = input("Enter text (or 'quit'): ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            audio = input("Enter audio file path (optional): ").strip()
            if not audio:
                audio = None
            elif not os.path.exists(audio):
                print(f"⚠️ Audio file not found: {audio}")
                audio = None
            
            image = input("Enter image file path (optional): ").strip()
            if not image:
                image = None
            elif not os.path.exists(image):
                print(f"⚠️ Image file not found: {image}")
                image = None
            
            if not any([text, audio, image]):
                print("⚠️ Please provide at least one input modality")
                continue
            
            try:
                result = predictor.predict_sentiment(
                    text=text if text else None,
                    audio_path=audio,
                    image_path=image
                )
                predictor.print_results(result, text, audio, image)
            except Exception as e:
                print(f"❌ Error during prediction: {e}")
    
    else:
        # Command line mode
        if not any([args.text, args.audio, args.image]):
            print("❌ Please provide at least one input (--text, --audio, or --image)")
            print("   Or use --interactive for interactive mode")
            return
        
        try:
            result = predictor.predict_sentiment(
                text=args.text,
                audio_path=args.audio,
                image_path=args.image
            )
            predictor.print_results(result, args.text, args.audio, args.image)
        except Exception as e:
            print(f"❌ Error during prediction: {e}")

if __name__ == "__main__":
    main()
