# End-to-End Multi-Modal Sentiment Analysis Training

This document explains the comprehensive end-to-end training approach for the multi-modal sentiment analysis system using real-world pre-trained models.

## Overview

Instead of using a hybrid approach with frozen pre-trained models, this implementation provides proper fine-tuning of all components:

- **Text**: RoBERTa (fine-tuned for sentiment analysis)
- **Audio**: Wav2Vec2 (fine-tuned for audio sentiment)  
- **Visual**: ViT (fine-tuned for visual sentiment)
- **Fusion**: Attention-based multi-modal fusion layer

## Architecture

### EndToEndMultiModalModel

The main model class that integrates all modalities:

```python
class EndToEndMultiModalModel(nn.Module):
    def __init__(self, num_classes=3):
        # Individual pre-trained models
        self.text_model = TextSentimentModel(num_labels=num_classes)
        self.audio_model = AudioSentimentModel(num_classes=num_classes) 
        self.visual_model = VisualSentimentModel(num_classes=num_classes)
        
        # Fusion and attention layers
        self.fusion = nn.Sequential(...)
        self.attention = nn.MultiheadAttention(...)
```

### Individual Model Components

#### 1. TextSentimentModel (RoBERTa)
- **Base Model**: `roberta-base`
- **Fine-tuning**: Yes, with optional freezing
- **Input**: Tokenized text sequences
- **Output**: 3-class sentiment predictions

#### 2. AudioSentimentModel (Wav2Vec2)
- **Base Model**: `facebook/wav2vec2-base-960h`
- **Fine-tuning**: Yes, with optional freezing
- **Input**: Raw audio waveforms (16kHz)
- **Output**: 3-class sentiment predictions

#### 3. VisualSentimentModel (ViT)
- **Base Model**: `google/vit-base-patch16-224`
- **Fine-tuning**: Yes, with optional freezing
- **Input**: RGB images (224x224)
- **Output**: 3-class sentiment predictions

## Training Strategy

### 1. Differential Learning Rates

The training uses different learning rates for different components:

```python
param_groups = [
    # Pre-trained model parameters (lower learning rate)
    {
        'params': [...],  # RoBERTa, Wav2Vec2, ViT parameters
        'lr': learning_rate / 10  # 10x lower learning rate
    },
    # New/fusion parameters (higher learning rate)  
    {
        'params': [...],  # Fusion and attention layers
        'lr': learning_rate
    }
]
```

### 2. Gradual Unfreezing (Optional)

You can start with frozen pre-trained models and gradually unfreeze them:

```python
# Start with frozen models
text_model = TextSentimentModel(freeze_roberta=True)
audio_model = AudioSentimentModel(freeze_wav2vec2=True)
visual_model = VisualSentimentModel(freeze_vit=True)

# Later unfreeze for fine-tuning
for param in model.text_model.roberta.parameters():
    param.requires_grad = True
```

### 3. Learning Rate Scheduling

Uses warmup and linear decay:

```python
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

## Usage

### 1. Test the Setup

First, run the test suite to ensure everything is configured correctly:

```bash
python tests/test_end_to_end_training.py
```

### 2. Start Training

Run the end-to-end training:

```bash
python training/train_end_to_end.py \
    --epochs 10 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --warmup_steps 1000 \
    --device cpu
```

### 3. Monitor Training

The training script will:
- Save the best model based on validation accuracy
- Log training progress to `logs/training_end_to_end.log`
- Save training history to `models/training_history_end_to_end.json`
- Create checkpoints every 5 epochs

## Data Requirements

The training expects the following data structure:

```python
batch = {
    'text': ["Text sample 1", "Text sample 2", ...],        # List of strings
    'audio_path': ["/path/to/audio1.wav", "/path/to/audio2.wav", ...],  # List of file paths
    'image': [PIL.Image, PIL.Image, ...],                   # List of PIL Images
    'label': torch.tensor([0, 1, 2, ...])                   # Class labels (0=negative, 1=neutral, 2=positive)
}
```

## Model Outputs

### Training Mode
```python
outputs = model(
    text_inputs=text_inputs,
    audio_inputs=audio_inputs,
    visual_inputs=visual_inputs,
    text_attention_mask=text_attention_mask
)
# outputs.shape: [batch_size, num_classes]
```

### Inference Mode with Attention
```python
outputs, attention_weights = model(
    text_inputs=text_inputs,
    audio_inputs=audio_inputs,
    visual_inputs=visual_inputs,
    text_attention_mask=text_attention_mask,
    return_attention=True
)
# attention_weights.shape: [batch_size, num_heads, num_modalities, num_modalities]
```

## Performance Optimization

### 1. Memory Optimization
- Use gradient checkpointing for large models
- Implement mixed precision training (FP16)
- Use gradient accumulation for larger effective batch sizes

### 2. Speed Optimization
- Pre-process and cache audio features
- Use DataLoader with multiple workers
- Enable torch.compile for faster training (PyTorch 2.0+)

### 3. Model Size Optimization
- Use DistilRoBERTa instead of RoBERTa-base
- Use smaller ViT variants (vit-small)
- Implement knowledge distillation

## Evaluation Metrics

The training tracks:
- **Accuracy**: Overall classification accuracy
- **Loss**: Cross-entropy loss
- **Per-modality Performance**: Individual modality contributions
- **Attention Weights**: Modality importance for interpretability

## Saved Models

After training, you'll have:
- `models/end_to_end_multimodal_best.pth`: Best model (highest validation accuracy)
- `models/end_to_end_multimodal_epoch_X.pth`: Periodic checkpoints
- `models/training_history_end_to_end.json`: Training metrics history

## Integration with Inference

The trained model can be used with the existing inference scripts:

### Streamlit App
```bash
streamlit run app/streamlit_app.py
```

### CLI Interface
```bash
python app/enhanced_predict.py --interactive
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 4`
   - Enable gradient checkpointing
   - Use CPU: `--device cpu`

2. **Slow Training**
   - Increase number of DataLoader workers
   - Use smaller models or freeze more layers
   - Pre-compute and cache features

3. **Poor Convergence**
   - Adjust learning rates
   - Increase warmup steps
   - Check data quality and class balance

### Debugging

Enable detailed logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

Monitor GPU usage:
```bash
nvidia-smi --loop=1
```

## Next Steps

1. **Hyperparameter Tuning**: Use Optuna or similar for systematic tuning
2. **Advanced Fusion**: Implement cross-modal attention mechanisms
3. **Self-Supervised Pre-training**: Use unlabeled multi-modal data
4. **Model Compression**: Implement pruning and quantization
5. **Deployment**: Package for production serving

This end-to-end approach provides a solid foundation for high-quality multi-modal sentiment analysis with proper fine-tuning of all components.
