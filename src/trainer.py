import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class MultiModalTrainer:
    """Trainer class for multi-modal sentiment analysis"""
    
    def __init__(self, model, device='cpu', learning_rate=1e-3):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        for batch in train_loader:
            # Move data to device
            text_features = batch['text_features'].to(self.device)
            audio_features = batch['audio_features'].to(self.device)
            visual_features = batch['visual_features'].to(self.device)
            labels = batch['sentiment_label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs, attention_weights = self.model(text_features, audio_features, visual_features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Collect predictions for accuracy
            predictions = torch.argmax(outputs, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_attention_weights = {'text': [], 'audio': [], 'visual': []}
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                text_features = batch['text_features'].to(self.device)
                audio_features = batch['audio_features'].to(self.device)
                visual_features = batch['visual_features'].to(self.device)
                labels = batch['sentiment_label'].to(self.device)
                
                # Forward pass
                outputs, attention_weights = self.model(text_features, audio_features, visual_features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # Collect predictions
                predictions = torch.argmax(outputs, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Collect attention weights
                all_attention_weights['text'].extend(attention_weights['text_attention'].cpu().numpy())
                all_attention_weights['audio'].extend(attention_weights['audio_attention'].cpu().numpy())
                all_attention_weights['visual'].extend(attention_weights['visual_attention'].cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy, all_predictions, all_labels, all_attention_weights
    
    def train(self, train_loader, val_loader, num_epochs=10):
        """Train the model for multiple epochs"""
        print("Starting training...")
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, _, _, _ = self.validate(val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print("-" * 50)
    
    def evaluate(self, test_loader):
        """Evaluate the model on test data"""
        print("Evaluating on test data...")
        
        test_loss, test_acc, predictions, labels, attention_weights = self.validate(test_loader)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Classification report
        class_names = ['Negative', 'Neutral', 'Positive']
        print("\nClassification Report:")
        print(classification_report(labels, predictions, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return test_acc, attention_weights
    
    def plot_training_history(self):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def analyze_attention(self, attention_weights):
        """Analyze attention weights across modalities"""
        text_attn = np.mean(attention_weights['text'])
        audio_attn = np.mean(attention_weights['audio'])
        visual_attn = np.mean(attention_weights['visual'])
        
        print(f"\nAverage Attention Weights:")
        print(f"Text: {text_attn:.3f}")
        print(f"Audio: {audio_attn:.3f}")
        print(f"Visual: {visual_attn:.3f}")
        
        # Plot attention distribution
        modalities = ['Text', 'Audio', 'Visual']
        weights = [text_attn, audio_attn, visual_attn]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(modalities, weights, color=['skyblue', 'lightgreen', 'salmon'])
        plt.title('Average Attention Weights by Modality')
        plt.ylabel('Attention Weight')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, weight in zip(bars, weights):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{weight:.3f}', ha='center', va='bottom')
        
        plt.show()
        
        return {'text': text_attn, 'audio': audio_attn, 'visual': visual_attn}
