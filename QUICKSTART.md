# 🎯 Quick Start Guide

## Installation

```bash
# Clone/navigate to directory
cd "Multi Modal-Sentiment Analysis"

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Option 1: Direct Testing (No Training Required)
```bash
# Test the architecture with pretrained models
python test_new_architecture.py

# Launch web app (works without training)
streamlit run app.py
```

### Option 2: Full Pipeline (Training + Deployment)
```bash
# Step 1: Train the model
python train.py

# Step 2: Launch web app
streamlit run app.py

# Navigate to "Training Info" page to see curves
```

## Web App Pages

1. **🔮 Prediction**
   - Enter text, upload audio/image
   - Click "Analyze Sentiment"
   - See prediction + attention visualization

2. **📊 Training Info**
   - View training curves (if trained)
   - See final metrics

3. **ℹ️ About**
   - Learn about the architecture
   - Understand use cases

## Project Components

| File | Purpose |
|------|---------|
| `train.py` | Train the model |
| `app.py` | Streamlit web interface |
| `test_new_architecture.py` | Quick system test |
| `src/models/` | Model architecture |
| `src/data/` | Data loading |
| `src/training/` | Training pipeline |
| `src/pipeline/` | Inference |

## Common Issues

**Issue**: "No module named 'src'"
- **Fix**: Make sure you're in the project root directory

**Issue**: "CUDA out of memory"
- **Fix**: Training uses CPU by default. If you enabled GPU and see this, reduce batch_size in `train.py`

**Issue**: "No checkpoint found"
- **Fix**: This is normal if you haven't trained. The app works with untrained models too.

## Tips

- The system works out-of-the-box without training (uses pretrained encoders)
- Training improves the fusion layer specifically for sentiment analysis
- Dummy data is auto-generated if no real data exists
- Check `models/training_history.json` after training for detailed metrics
