import torch
import os
import sys
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.fusion_model import MultiModalSentimentAnalyzer

class MultiModalPredictor:
    """Inference class for trained multi-modal sentiment analysis model"""
    
    def __init__(self, model_path='multimodal_sentiment_model.pth', device='cpu'):
        self.device = device
        
        # Initialize model
        self.model = MultiModalSentimentAnalyzer(
            input_dim=768,
            hidden_dim=256,
            fusion_dim=128,
            num_classes=3
        )
        
        # Load trained weights
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ Model loaded from {model_path}")
        else:
            print(f"⚠️ Model file {model_path} not found. Using untrained model.")
        
        self.model.to(device)
        self.model.eval()
        
        # Sentiment mapping
        self.sentiment_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        
    def predict_from_features(self, text_features, audio_features, visual_features):
        """Predict sentiment from pre-extracted features"""
        
        # Convert to tensors if needed
        if not isinstance(text_features, torch.Tensor):
            text_features = torch.FloatTensor(text_features)
        if not isinstance(audio_features, torch.Tensor):
            audio_features = torch.FloatTensor(audio_features)
        if not isinstance(visual_features, torch.Tensor):
            visual_features = torch.FloatTensor(visual_features)
        
        # Add batch dimension if needed
        if text_features.dim() == 1:
            text_features = text_features.unsqueeze(0)
            audio_features = audio_features.unsqueeze(0)
            visual_features = visual_features.unsqueeze(0)
        
        # Move to device
        text_features = text_features.to(self.device)
        audio_features = audio_features.to(self.device)
        visual_features = visual_features.to(self.device)
        
        # Predict
        result = self.model.predict_sentiment(text_features, audio_features, visual_features)
        
        return {
            'sentiment_label': result['predictions'][0],
            'sentiment_name': self.sentiment_map[result['predictions'][0]],
            'confidence': torch.max(result['probabilities']).item(),
            'probabilities': {
                'negative': result['probabilities'][0][0].item(),
                'neutral': result['probabilities'][0][1].item(),
                'positive': result['probabilities'][0][2].item()
            },
            'attention_weights': {
                'text': result['attention_weights']['text_attention'][0].item(),
                'audio': result['attention_weights']['audio_attention'][0].item(),
                'visual': result['attention_weights']['visual_attention'][0].item()
            }
        }
    
    def predict_from_files(self, text_file, audio_file, visual_file):
        """Predict sentiment from feature files"""
        
        # Load features
        text_features = np.load(text_file)
        audio_features = np.load(audio_file)
        visual_features = np.load(visual_file)
        
        return self.predict_from_features(text_features, audio_features, visual_features)
    
    def analyze_sample(self, sample_id, features_dir='data/sample_multimodal/features'):
        """Analyze a sample from the dataset"""
        
        text_file = os.path.join(features_dir, 'text', f'{sample_id}.npy')
        audio_file = os.path.join(features_dir, 'audio', f'{sample_id}.npy')
        visual_file = os.path.join(features_dir, 'visual', f'{sample_id}.npy')
        
        if not all(os.path.exists(f) for f in [text_file, audio_file, visual_file]):
            raise FileNotFoundError(f"Feature files for {sample_id} not found")
        
        result = self.predict_from_files(text_file, audio_file, visual_file)
        
        print(f"\n🔮 Analysis for {sample_id}:")
        print(f"Predicted Sentiment: {result['sentiment_name']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"\nProbability Distribution:")
        print(f"  Negative: {result['probabilities']['negative']:.3f}")
        print(f"  Neutral:  {result['probabilities']['neutral']:.3f}")
        print(f"  Positive: {result['probabilities']['positive']:.3f}")
        print(f"\nAttention Weights:")
        print(f"  Text:   {result['attention_weights']['text']:.3f}")
        print(f"  Audio:  {result['attention_weights']['audio']:.3f}")
        print(f"  Visual: {result['attention_weights']['visual']:.3f}")
        
        return result

def demo_predictions():
    """Demo function to show predictions on sample data"""
    
    print("🚀 Multi-Modal Sentiment Analysis - Inference Demo")
    print("=" * 60)
    
    # Initialize predictor
    predictor = MultiModalPredictor(device='cpu')
    
    # Analyze all sample data
    sample_ids = [f'sample_{i:03d}' for i in range(10)]
    
    for sample_id in sample_ids:
        try:
            result = predictor.analyze_sample(sample_id)
        except FileNotFoundError:
            print(f"⚠️ Files for {sample_id} not found, skipping...")
            continue
        print("-" * 60)
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    demo_predictions()
