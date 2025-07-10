"""n the forward method of EndToEndMultiModalModel class
End-to-End Multi-Modal Sentiment Analysis Training
Properly fine-tunes all pre-trained models (RoBERTa, Wav2Vec2, ViT) for sentiment analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
from pathlib import Path
import sys
import argparse
from datetime import datetime
import numpy as np
import os
import tempfile
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from real_data_loader import get_data_config, create_real_data_loaders
from feature_extractors import TextSentimentModel, AudioSentimentModel, VisualSentimentModel
from models.fusion_model import MultiModalSentimentAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training_end_to_end.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EndToEndMultiModalModel(nn.Module):
    """
    End-to-end multi-modal sentiment analysis model that fine-tunes all components
    """
    def __init__(self, num_classes=3):
        super().__init__()
        
        # Initialize individual models (these will be fine-tuned)
        # The sub-models are responsible for their own preprocessing now
        self.text_model = TextSentimentModel(num_labels=num_classes)
        self.audio_model = AudioSentimentModel(num_classes=num_classes)
        self.visual_model = VisualSentimentModel(num_classes=num_classes)
        
        # Fusion layer for combining modalities
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 3, 256),  # 3 modalities * num_classes each
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Attention mechanism for modality weighting
        self.attention = nn.MultiheadAttention(
            embed_dim=num_classes,
            num_heads=1,
            batch_first=True
        )
        
        self.num_classes = num_classes
        
    def forward(self, texts=None, raw_audio=None, raw_images=None, return_attention=False):
        """
        Forward pass through the end-to-end model.
        This method now accepts raw data and passes it to the appropriate sub-models,
        which handle their own preprocessing.
        
        Args:
            texts (List[str]): A list of raw text strings.
            raw_audio (List[np.ndarray]): A list of raw audio waveforms.
            raw_images (List[PIL.Image]): A list of raw images.
            return_attention (bool): Whether to return attention weights.
        """
        
        modality_outputs = []
        modality_names = []
        
        # Get the current device from the model's parameters
        self.device = next(self.parameters()).device

        # Process text modality
        if texts is not None:
            if self.text_model.model_loaded:
                # Tokenize and process text
                encoded = self.text_model.tokenizer(
                    texts, truncation=True, padding=True, 
                    return_tensors='pt', max_length=512
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                text_logits = self.text_model(encoded['input_ids'], encoded['attention_mask'])
                text_probs = torch.softmax(text_logits, dim=-1)
            else:
                batch_size = len(texts)
                text_probs = torch.ones(batch_size, self.num_classes, device=self.device) / self.num_classes
            
            modality_outputs.append(text_probs)
            modality_names.append('text')
        
        # Process audio modality  
        if raw_audio is not None:
            if self.audio_model.use_wav2vec2:
                # The audio model's forward pass handles processing the list of raw waveforms
                audio_logits = self.audio_model(raw_audio)
                audio_probs = torch.softmax(audio_logits, dim=-1)
            else:
                batch_size = len(raw_audio)
                audio_probs = torch.ones(batch_size, self.num_classes, device=self.device) / self.num_classes
            
            modality_outputs.append(audio_probs)
            modality_names.append('audio')
        
        # Process visual modality
        if raw_images is not None:
            if self.visual_model.use_vit:
                # The visual model's forward pass handles processing the list of PIL images
                visual_logits = self.visual_model(raw_images)
                visual_probs = torch.softmax(visual_logits, dim=-1)
            else:
                batch_size = len(raw_images)
                visual_probs = torch.ones(batch_size, self.num_classes, device=self.device) / self.num_classes
            
            modality_outputs.append(visual_probs)
            modality_names.append('visual')
        
        if not modality_outputs:
            raise ValueError("At least one modality input must be provided")
        
        # Stack modality outputs for attention
        modality_stack = torch.stack(modality_outputs, dim=1)
        
        # Apply attention mechanism
        attended_features, attention_weights = self.attention(
            modality_stack, modality_stack, modality_stack
        )
        
        # Flatten for fusion layer
        fusion_input = attended_features.flatten(start_dim=1)
        
        # Final prediction through fusion layer
        final_output = self.fusion(fusion_input)
        
        if return_attention:
            return final_output, attention_weights
        return final_output
 

class EndToEndTrainer:
    """
    Trainer for end-to-end multi-modal sentiment analysis
    """
    def __init__(self, model, device='cpu', learning_rate=2e-5, warmup_steps=1000):
        self.model = model
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Setup optimizer with different learning rates for different components
        param_groups = [
            # Pre-trained model parameters (lower learning rate)
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(pretrained in n for pretrained in ['roberta', 'wav2vec2', 'vit'])],
                'lr': learning_rate / 10  # 10x lower learning rate for pre-trained parts
            },
            # New/fusion parameters (higher learning rate)
            {
                'params': [p for n, p in model.named_parameters() 
                          if not any(pretrained in n for pretrained in ['roberta', 'wav2vec2', 'vit'])],
                'lr': learning_rate
            }
        ]
        
        self.optimizer = optim.AdamW(param_groups, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = None
        self.warmup_steps = warmup_steps
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def setup_scheduler(self, total_steps):
        """Setup learning rate scheduler"""
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with end-to-end fine-tuning"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Data is already in raw format from the collate function
                texts = batch['texts']
                raw_audio = batch['audios']
                raw_images = batch['images']
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                outputs = self.model(
                    texts=texts,
                    raw_audio=raw_audio, 
                    raw_images=raw_images
                )
                
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                if batch_idx % 10 == 0:
                    logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                              f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
                              
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
            
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        
        logger.info(f'Train Epoch {epoch}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, epoch):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    # Data is already in raw format from the collate function
                    texts = batch['texts']
                    raw_audio = batch['audios']
                    raw_images = batch['images']
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        texts=texts,
                        raw_audio=raw_audio,
                        raw_images=raw_images
                    )
                    
                    loss = self.criterion(outputs, labels)
                    
                    # Statistics
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_acc)
        
        logger.info(f'Validation Epoch {epoch}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        return epoch_loss, epoch_acc
    
    def save_model(self, filepath, epoch, best_val_acc):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='End-to-End Multi-Modal Sentiment Analysis Training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--save_dir', type=str, default='../models', help='Directory to save models')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps for scheduler')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    try:
        config = get_data_config()
        train_loader, val_loader, test_loader = create_real_data_loaders(
            config, 
            batch_size=args.batch_size
        )
        logger.info(f"Data loaded successfully. Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Initialize model
    logger.info("Initializing end-to-end model...")
    model = EndToEndMultiModalModel(num_classes=3)
    
    # Initialize trainer
    trainer = EndToEndTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps
    )
    
    # Setup scheduler
    total_steps = len(train_loader) * args.epochs
    trainer.setup_scheduler(total_steps)
    
    # Training loop
    logger.info("Starting end-to-end training...")
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n=== Epoch {epoch}/{args.epochs} ===")
        
        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
        
        # Validate
        val_loss, val_acc = trainer.validate(val_loader, epoch)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = save_dir / 'end_to_end_multimodal_best.pth'
            trainer.save_model(best_model_path, epoch, best_val_acc)
            logger.info(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_path = save_dir / f'end_to_end_multimodal_epoch_{epoch}.pth'
            trainer.save_model(checkpoint_path, epoch, best_val_acc)
    
    # Final evaluation on test set
    logger.info("\n=== Final Test Evaluation ===")
    test_loss, test_acc = trainer.validate(test_loader, "Test")
    logger.info(f"Final Test Accuracy: {test_acc:.2f}%")
    
    # Save training history
    history = {
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'train_accuracies': trainer.train_accuracies,
        'val_accuracies': trainer.val_accuracies,
        'best_val_acc': best_val_acc,
        'final_test_acc': test_acc
    }
    
    history_path = save_dir / 'training_history_end_to_end.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Final test accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
