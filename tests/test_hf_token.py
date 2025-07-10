#!/usr/bin/env python3
"""
Test script to verify HuggingFace token loading from .env file
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def test_hf_token_loading():
    """Test that HF_TOKEN is properly loaded from .env file"""
    
    print("🧪 Testing HuggingFace token loading...")
    print("=" * 50)
    
    # Test 1: Direct environment variable loading
    import os
    from dotenv import load_dotenv
    
    env_path = Path(__file__).parent.parent / '.env'
    print(f"📁 Loading .env from: {env_path}")
    
    if env_path.exists():
        load_dotenv(env_path)
        token = os.getenv('HF_TOKEN', '')
        if token:
            # Only show first 10 and last 5 characters for security
            masked_token = f"{token[:10]}...{token[-5:]}"
            print(f"✅ HF_TOKEN loaded successfully: {masked_token}")
        else:
            print("❌ HF_TOKEN not found in .env file")
    else:
        print(f"❌ .env file not found at {env_path}")
    
    print()
    
    # Test 2: Import feature_extractors to test automatic loading
    print("🧪 Testing automatic loading via feature_extractors...")
    try:
        from feature_extractors import HF_TOKEN
        if HF_TOKEN:
            masked_token = f"{HF_TOKEN[:10]}...{HF_TOKEN[-5:]}"
            print(f"✅ HF_TOKEN loaded via feature_extractors: {masked_token}")
        else:
            print("❌ HF_TOKEN not loaded via feature_extractors")
    except ImportError as e:
        print(f"❌ Failed to import feature_extractors: {e}")
    
    print()
    
    # Test 3: Test utility function
    print("🧪 Testing utility function...")
    try:
        from utils import load_hf_token, setup_hf_authentication
        token = load_hf_token()
        if token:
            masked_token = f"{token[:10]}...{token[-5:]}"
            print(f"✅ HF_TOKEN loaded via utils: {masked_token}")
            
            # Test authentication
            auth_success = setup_hf_authentication()
            if auth_success:
                print("✅ HuggingFace authentication successful")
            else:
                print("❌ HuggingFace authentication failed")
        else:
            print("❌ HF_TOKEN not loaded via utils")
    except ImportError as e:
        print(f"❌ Failed to import utils: {e}")
    
    print()
    print("🎉 HuggingFace token test completed!")

if __name__ == "__main__":
    test_hf_token_loading()
