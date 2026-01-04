import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    """Attention-based fusion mechanism for multi-modal features"""
    
    def __init__(self, feature_dim=128):
        super(AttentionFusion, self).__init__()
        self.feature_dim = feature_dim
        
        # Attention layers for each modality
        self.text_attention = nn.Linear(feature_dim, 1)
        self.audio_attention = nn.Linear(feature_dim, 1)
        self.visual_attention = nn.Linear(feature_dim, 1)
        
        # Final classification layer
        self.classifier = nn.Linear(feature_dim, 3)  # 3 classes: negative, neutral, positive
        
    def forward(self, text_features, audio_features, visual_features):
        # Compute attention weights
        text_attn = torch.sigmoid(self.text_attention(text_features))
        audio_attn = torch.sigmoid(self.audio_attention(audio_features))
        visual_attn = torch.sigmoid(self.visual_attention(visual_features))
        
        # Normalize attention weights
        total_attn = text_attn + audio_attn + visual_attn
        text_attn = text_attn / total_attn
        audio_attn = audio_attn / total_attn
        visual_attn = visual_attn / total_attn
        
        # Apply attention weights
        weighted_text = text_features * text_attn
        weighted_audio = audio_features * audio_attn
        weighted_visual = visual_features * visual_attn
        
        # Fuse features
        fused_features = weighted_text + weighted_audio + weighted_visual
        
        # Final classification
        output = self.classifier(fused_features)
        
        return output, {
            'text_attention': text_attn,
            'audio_attention': audio_attn,
            'visual_attention': visual_attn
        }

class MultiModalSentimentAnalyzer(nn.Module):
    """Complete multi-modal sentiment analysis system with attention fusion"""
    
    def __init__(self, input_dim=768, hidden_dim=256, fusion_dim=128, num_classes=3):
        super(MultiModalSentimentAnalyzer, self).__init__()
        
        # Individual modality encoders
        self.text_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, fusion_dim),
            nn.ReLU()
        )
        
        self.audio_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, fusion_dim),
            nn.ReLU()
        )
        
        self.visual_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, fusion_dim),
            nn.ReLU()
        )
        
        # Attention fusion module
        self.fusion = AttentionFusion(fusion_dim)
        
    def forward(self, text_features, audio_features, visual_features):
        # Encode each modality
        text_encoded = self.text_encoder(text_features)
        audio_encoded = self.audio_encoder(audio_features)
        visual_encoded = self.visual_encoder(visual_features)
        
        # Fuse with attention
        output, attention_weights = self.fusion(text_encoded, audio_encoded, visual_encoded)
        
        return output, attention_weights
    
    def predict_sentiment(self, text_features, audio_features, visual_features):
        """Predict sentiment with class labels"""
        self.eval()
        with torch.no_grad():
            logits, attention_weights = self.forward(text_features, audio_features, visual_features)
            probabilities = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)
            
            # Convert back to sentiment labels (-1, 0, 1)
            sentiment_map = {0: -1, 1: 0, 2: 1}  # negative, neutral, positive
            sentiment_labels = [sentiment_map[pred.item()] for pred in predicted_class]
            
            return {
                'predictions': sentiment_labels,
                'probabilities': probabilities,
                'attention_weights': attention_weights
            }
