import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path

def plot_training_history(history_path: Path):
    """Plot training history from JSON file"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', marker='o')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', marker='o')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_attention_heatmap(attention_weights, modality_names=['Text', 'Audio', 'Visual']):
    """
    Visualize attention weights as a heatmap
    
    Args:
        attention_weights: (3, 3) or (batch, 3, 3) array
        modality_names: Names of modalities
    """
    # Handle batch dimension
    if attention_weights.ndim == 3:
        attention_weights = attention_weights.mean(axis=0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        attention_weights,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        xticklabels=modality_names,
        yticklabels=modality_names,
        ax=ax,
        cbar_kws={'label': 'Attention Weight'},
        vmin=0,
        vmax=1
    )
    
    ax.set_title('Cross-Modal Attention Weights', fontsize=14, fontweight='bold')
    ax.set_xlabel('Attended To', fontsize=12)
    ax.set_ylabel('Attending From', fontsize=12)
    
    return fig

def plot_modality_importance(attention_weights, modality_names=['Text', 'Audio', 'Visual']):
    """Plot bar chart of modality importance"""
    # Average attention each modality receives
    if attention_weights.ndim == 3:
        attention_weights = attention_weights.mean(axis=0)
    
    importance = attention_weights.mean(axis=0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax.bar(modality_names, importance, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Average Attention Weight', fontsize=12)
    ax.set_title('Modality Importance Analysis', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(importance) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    return fig
