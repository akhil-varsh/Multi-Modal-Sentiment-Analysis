# Multi-Modal Sentiment Analysis System

A ready-to-use system for analyzing sentiment from text, audio, and visual data using pre-trained transformer models (RoBERTa, ViT) and attention-based fusion. **Works out-of-the-box without fine-tuning!**

## ✨ Key Features

- 🔥 **Ready to Use**: Works immediately with pre-trained models
- 🤖 **Multi-Modal**: Analyzes text, audio, and visual data
- 🚀 **CPU-Friendly**: Runs on CPU, no GPU required
- 🔧 **Modular Design**: Easy to extend and customize
- 📊 **Attention Fusion**: Intelligent combination of modalities

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Wav2Vec2 Integration
```bash
python tests/test_wav2vec2_integration.py
```

### 3. Run Comprehensive Demo
```bash
python tests/wav2vec2_demo.py
```

### 4. Interactive Web Interface
```bash
streamlit run app/streamlit_app.py
```

### 5. Command Line Interface
```bash
python app/enhanced_predict.py --interactive
```

## 🏗️ Project Structure

```
Multi Modal-Sentiment Analysis/
├── configs/
│   └── config.py              # Configuration settings
├── data/
│   └── sample_multimodal/     # Sample dataset
│       ├── sample_multimodal_data.csv
│       └── features/
│           ├── audio/         # Audio feature files (.npy)
│           ├── text/          # Text feature files (.npy)
│           └── visual/        # Visual feature files (.npy)
├── notebooks/                 # Jupyter notebooks for exploration
├── scripts/
│   ├── download_data.py       # Dataset download and setup
│   ├── create_custom_dataset.py  # Custom dataset creation
│   ├── train_multimodal.py    # Main training script
│   └── predict.py             # Inference script
├── src/
│   ├── data_loader.py         # Data loading utilities
│   ├── trainer.py             # Training pipeline
│   ├── models/
│   │   ├── individual_models.py  # Text, Audio, Visual models
│   │   └── fusion_model.py    # Attention fusion network
│   └── __init__.py
├── utils/
│   ├── data_utils.py          # Data processing utilities
│   └── helpers.py             # Helper functions
├── models/                    # Saved model checkpoints
├── results/                   # Training results and plots
├── logs/                      # Training logs
└── requirements.txt
```

## 🚀 Features

### Individual Modality Models
- **Text Analysis**: RoBERTa-based sentiment classification
- **Audio Analysis**: **Wav2Vec2** feature extraction with sentiment classification head
- **Visual Analysis**: Vision Transformer (ViT) for image sentiment

#### Audio Processing with Wav2Vec2
The system uses Facebook's Wav2Vec2 model for sophisticated audio feature extraction:
- **Model**: `facebook/wav2vec2-base-960h` (768-dimensional features)
- **Fallback**: MFCC features if Wav2Vec2 unavailable
- **Processing**: Automatic resampling to 16kHz, 30-second clips
- **Classification**: Custom neural network head for sentiment prediction

### Fusion Network
- **Attention Mechanism**: Learns to weight modalities dynamically
- **Multi-Head Attention**: Sophisticated fusion strategy
- **Learnable Weights**: Adapts to different input types

### Training Pipeline
- **End-to-End Training**: Complete training and evaluation pipeline
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Optimized training
- **Comprehensive Logging**: Detailed training metrics

### Evaluation & Visualization
- **Attention Analysis**: Visualize which modalities are important
- **Training Curves**: Monitor training progress
- **Confusion Matrix**: Detailed performance analysis
- **Model Checkpointing**: Save best models automatically

## 📊 Dataset

The system works with multimodal datasets containing:
- **Text**: Sentiment-bearing text content (SST-2 dataset)
- **Audio**: Speech or audio files with emotional content (EMO-DB dataset)
- **Visual**: Images with facial expressions (FER-2013 dataset)
- **Labels**: Sentiment labels (-1: negative, 0: neutral, 1: positive)

### Supported Datasets
- **SST-2**: Stanford Sentiment Treebank for text sentiment
- **EMO-DB**: Berlin Database of Emotional Speech for audio
- **FER-2013**: Facial Expression Recognition dataset for visual sentiment
- **Custom Datasets**: Create your own using provided tools

### Audio Processing Features
- **Wav2Vec2 Integration**: State-of-the-art audio feature extraction
- **Automatic Fallback**: MFCC features when Wav2Vec2 unavailable
- **Real-time Processing**: Optimized for live audio input
- **Format Support**: WAV, MP3, and other common audio formats

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd "Multi Modal-Sentiment Analysis"
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup dataset**
```bash
python scripts/download_data.py
```

## 🎯 Quick Start

### 1. Train the Model
```bash
python scripts/train_multimodal.py
```

### 2. Make Predictions
```bash
python scripts/predict.py
```

### 3. Analyze Results
Results will be saved in the `results/` directory including:
- Training curves
- Attention weight analysis
- Confusion matrix
- Model checkpoints

## 📈 Model Architecture

### Individual Encoders
```python
Text Encoder: Linear(768) -> ReLU -> Dropout -> Linear(256)
Audio Encoder: Linear(768) -> ReLU -> Dropout -> Linear(256)  
Visual Encoder: Linear(768) -> ReLU -> Dropout -> Linear(256)
```

### Attention Fusion
```python
MultiHeadAttention(256) -> Linear(128) -> Classifier(3)
```

### Output
- **3 Classes**: Negative, Neutral, Positive
- **Attention Weights**: Importance of each modality
- **Confidence Scores**: Prediction confidence

## 🔧 Configuration

Modify `configs/config.py` to customize:
- **Model architecture** (dimensions, layers)
- **Training parameters** (learning rate, epochs)
- **Data paths** and preprocessing
- **Evaluation settings**

## 📊 Results

The system provides comprehensive evaluation:

### Performance Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class performance
- **Confusion Matrix**: Detailed error analysis

### Attention Analysis
- **Modality Importance**: Which inputs matter most
- **Dynamic Weighting**: How attention changes per sample
- **Visualization**: Clear attention weight plots

## 🔬 Advanced Usage

### Custom Dataset Creation
```python
from scripts.create_custom_dataset import CustomDatasetCreator
creator = CustomDatasetCreator()
creator.create_sample_dataset()
```

### Model Inference
```python
from scripts.predict import MultiModalPredictor
predictor = MultiModalPredictor('models/best_model.pth')
result = predictor.predict(text_features, audio_features, visual_features)
```

### Attention Visualization
```python
from utils.helpers import plot_attention_analysis
plot_attention_analysis(attention_weights, save_path='attention_plot.png')
```

## 🎓 Research & Development

This system implements state-of-the-art multimodal fusion techniques:
- **Late Fusion**: Combines high-level representations
- **Attention Mechanism**: Learns optimal modality weighting
- **Transformer Architecture**: Leverages pre-trained models

### Key Papers
- RoBERTa: Liu et al. (2019)
- Wav2Vec2: Baevski et al. (2020)
- Vision Transformer: Dosovitskiy et al. (2021)
- Multimodal Fusion: Zadeh et al. (2017)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- CMU MultiComp Lab for multimodal datasets
- Hugging Face for transformer implementations
- PyTorch team for the deep learning framework
