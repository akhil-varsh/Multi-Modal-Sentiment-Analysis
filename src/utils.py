"""
Utility functions for the multi-modal sentiment analysis project
"""

import os
from pathlib import Path

def load_hf_token():
    """
    Load HuggingFace token from environment variables or .env file
    
    Returns:
        str: HuggingFace token or empty string if not found
    """
    # First try to get from environment variables
    hf_token = os.getenv('HF_TOKEN', '')
    
    if not hf_token:
        # Try loading from .env file
        try:
            from dotenv import load_dotenv
            # Load .env file from project root
            env_path = Path(__file__).parent.parent / '.env'
            load_dotenv(env_path)
            hf_token = os.getenv('HF_TOKEN', '')
        except ImportError:
            print("python-dotenv not installed. Install with: pip install python-dotenv")
    
    return hf_token

def setup_hf_authentication():
    """
    Setup HuggingFace authentication if token is available
    
    Returns:
        bool: True if authentication was successful, False otherwise
    """
    hf_token = load_hf_token()
    
    if hf_token:
        print("HuggingFace token found in environment")
        try:
            from huggingface_hub import login
            login(token=hf_token)
            print("✅ HuggingFace authentication successful")
            return True
        except Exception as e:
            print(f"❌ HuggingFace authentication failed: {e}")
            return False
    else:
        print("⚠️ No HuggingFace token found in environment variables")
        return False
