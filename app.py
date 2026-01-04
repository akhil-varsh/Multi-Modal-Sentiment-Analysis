import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import librosa
from pathlib import Path
import json

from src.pipeline.predictor import Predictor
from src.utils.visualization import (
    plot_attention_heatmap,
    plot_modality_importance,
    plot_training_history
)

# Page config
st.set_page_config(
    page_title="Multi-Modal Sentiment Analysis",
    page_icon="🎭",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    """Load the predictor (cached for performance)"""
    checkpoint_path = Path('models/best_model.pt')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = Predictor(
        checkpoint_path=checkpoint_path if checkpoint_path.exists() else None,
        device=device
    )
    
    return predictor

def main():
    st.markdown('<p class="big-font">🎭 Multi-Modal Sentiment Analysis</p>', unsafe_allow_html=True)
    st.markdown("### Understanding Human Emotions through Text, Audio, and Visual Inputs")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.radio("Select Page:", ["🔮 Prediction", "📊 Training Info", "ℹ️ About"])
        
        st.markdown("---")
        st.header("🛠️ System Info")
        
        checkpoint_exists = Path('models/best_model.pt').exists()
        history_exists = Path('models/training_history.json').exists()
        
        st.write(f"**Model Status:** {'✅ Trained' if checkpoint_exists else '⚠️ Untrained'}")
        st.write(f"**Training History:** {'✅ Available' if history_exists else '❌ None'}")
        st.write(f"**Device:** {'🚀 GPU' if torch.cuda.is_available() else '💻 CPU'}")
    
    # Main content based on page selection
    if page == "🔮 Prediction":
        show_prediction_page()
    elif page == "📊 Training Info":
        show_training_info_page()
    else:
        show_about_page()

def show_prediction_page():
    st.markdown("---")
    
    # Load predictor
    with st.spinner("Loading AI models..."):
        predictor = load_predictor()
    
    # Input section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📝 Text Input")
        text_input = st.text_area(
            "Enter text:",
            "I love this product! It's amazing!",
            height=100
        )
    
    with col2:
        st.subheader("🎤 Audio Input")
        audio_file = st.file_uploader("Upload audio (WAV/MP3)", type=['wav', 'mp3'])
        
        if audio_file is None:
            st.info("Using dummy audio for demo")
            audio_data = np.random.randn(16000).astype(np.float32)
        else:
            audio_bytes = audio_file.read()
            audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, duration=30)
            st.audio(audio_bytes)
    
    with col3:
        st.subheader("🖼️ Visual Input")
        image_file = st.file_uploader("Upload image (JPG/PNG)", type=['jpg', 'png', 'jpeg'])
        
        if image_file is None:
            st.info("Using dummy image for demo")
            image = Image.new('RGB', (224, 224), color='lightblue')
        else:
            image = Image.open(image_file)
        
        st.image(image, width=200)
    
    st.markdown("---")
    
    # Predict button
    if st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            try:
                # Run prediction
                result = predictor.predict_single(
                    text=text_input,
                    audio=audio_data,
                    image=image
                )
                
                sentiment = result['sentiment']
                probabilities = result['probabilities']
                attention_weights = result['attention_weights']
                
                # Display results
                st.success("Analysis Complete!")
                
                # Sentiment result
                st.markdown("### 🎯 Predicted Sentiment")
                
                if sentiment == "positive":
                    st.markdown(f"## :green[{sentiment.upper()}] 😊")
                elif sentiment == "negative":
                    st.markdown(f"## :red[{sentiment.upper()}] 😞")
                else:
                    st.markdown(f"## :orange[{sentiment.upper()}] 😐")
                
                # Probabilities
                st.markdown("### 📊 Confidence Scores")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Negative", f"{probabilities[0]:.1%}")
                with col2:
                    st.metric("Neutral", f"{probabilities[1]:.1%}")
                with col3:
                    st.metric("Positive", f"{probabilities[2]:.1%}")
                
                # Attention visualizations
                st.markdown("### 🔍 Attention Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Cross-Modal Attention:**")
                    fig1 = plot_attention_heatmap(attention_weights)
                    st.pyplot(fig1)
                
                with col2:
                    st.markdown("**Modality Importance:**")
                    fig2 = plot_modality_importance(attention_weights)
                    st.pyplot(fig2)
                
                st.info("""
                **Interpretation:** The heatmap shows how each modality attends to others. 
                The bar chart shows overall importance. Higher values indicate the model 
                relied more on that input type for the prediction.
                """)
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.exception(e)

def show_training_info_page():
    st.header("📊 Training Information")
    
    history_path = Path('models/training_history.json')
    
    if history_path.exists():
        # Load and display training history
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        st.success(f"Training completed for {len(history['train_loss'])} epochs")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Final Train Loss", f"{history['train_loss'][-1]:.4f}")
        with col2:
            st.metric("Final Val Loss", f"{history['val_loss'][-1]:.4f}")
        with col3:
            st.metric("Final Train Acc", f"{history['train_acc'][-1]:.2f}%")
        with col4:
            st.metric("Best Val Acc", f"{max(history['val_acc']):.2f}%")
        
        # Plot training curves
        st.markdown("### 📈 Training Curves")
        fig = plot_training_history(history_path)
        st.pyplot(fig)
        
    else:
        st.warning("No training history found. Please run `python train.py` first.")
        
        st.markdown("### 🚀 How to Train")
        st.code("""
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run training
python train.py

# 3. Models will be saved to models/
        """, language='bash')

def show_about_page():
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ## 🎯 Purpose
    This system combines **Text**, **Audio**, and **Visual** inputs to understand human emotions, 
    making it ideal for Human-Robot Interaction applications.
    
    ## 🧠 Architecture
    - **Text Encoder**: RoBERTa (understanding words)
    - **Audio Encoder**: Wav2Vec2 (detecting tone and emotion)
    - **Visual Encoder**: ViT (reading facial expressions)
    - **Fusion**: Multi-Head Attention (intelligent combination)
    
    ## 🔑 Key Features
    1. **Attention-Based Fusion**: Unlike simple averaging, the system learns which modality 
       to trust more based on the input.
    2. **Interpretability**: Attention weights show which inputs influenced the decision.
    3. **Robustness**: Can detect conflicting signals (e.g., sarcasm = positive words + negative tone).
    
    ## 📚 Use Cases
    - **Social Robots**: Understanding user emotions beyond just words
    - **Mental Health**: Detecting depression/anxiety from multimodal cues
    - **Customer Service**: Analyzing satisfaction from voice + text + expressions
    
    ## 🎓 Technical Details
    - **Framework**: PyTorch
    - **Pretrained Models**: HuggingFace Transformers
    - **Training**: Cross-Entropy Loss with AdamW optimizer
    - **Attention**: 8-head Multi-Head Self-Attention
    
    ## 📖 Citation
    If you use this system, please cite the foundational papers:
    - RoBERTa (Liu et al., 2019)
    - Wav2Vec2 (Baevski et al., 2020)
    - Vision Transformer (Dosovitskiy et al., 2021)
    """)

if __name__ == "__main__":
    main()
