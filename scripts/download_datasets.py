#!/usr/bin/env python3
"""
Download scripts for Text, Audio, and Image datasets
This script downloads and sets up:
1. Stanford Sentiment Treebank (SST-2) for text sentiment
2. EMO-DB for audio emotion 
3. FER-2013 for image emotion
"""

import os
import sys
import requests
import zipfile
import tarfile
import pandas as pd
from pathlib import Path
import shutil
from urllib.parse import urlparse
import datasets

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class DatasetDownloader:
    def __init__(self):
        self.data_dir = project_root / "data"
        self.create_directories()
    
    def create_directories(self):
        """Verify existing datasets and create necessary directories only for text."""
        # Only create text sentiment directory (for SST-2 download)
        text_dir = self.data_dir / "text_sentiment" / "sst2"
        text_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify existing datasets
        self.verify_existing_datasets()
        
        print("✓ Verified dataset structure")
    
    def verify_existing_datasets(self):
        """Verify that EMO-DB and FER-2013 datasets exist."""
        print("\n🔍 Verifying existing datasets...")
        
        # Check EMO-DB
        emo_db_path = self.data_dir / "audio_emotion" / "emodb" / "wav"
        if emo_db_path.exists():
            wav_files = list(emo_db_path.glob("*.wav"))
            print(f"✅ EMO-DB found: {len(wav_files)} audio files in {emo_db_path}")
        else:
            print(f"❌ EMO-DB not found at {emo_db_path}")
        
        # Check FER-2013
        fer_train_path = self.data_dir / "image_emotion" / "fer-2013" / "train"
        fer_test_path = self.data_dir / "image_emotion" / "fer-2013" / "test"
        
        if fer_train_path.exists() and fer_test_path.exists():
            # Count images in train folder
            train_count = sum(len(list(emotion_dir.glob("*.jpg"))) + len(list(emotion_dir.glob("*.png"))) 
                            for emotion_dir in fer_train_path.iterdir() if emotion_dir.is_dir())
            print(f"✅ FER-2013 found: ~{train_count} training images in {fer_train_path}")
        else:
            print(f"❌ FER-2013 not found at {fer_train_path} or {fer_test_path}")
    
    def download_file(self, url, destination, description="file"):
        """Download a file with progress indication."""
        print(f"Downloading {description}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r  Progress: {percent:.1f}%", end="", flush=True)
            
            print(f"\n✓ Downloaded {description}")
            return True
        except Exception as e:
            print(f"\n✗ Failed to download {description}: {e}")
            return False
    
    def extract_archive(self, archive_path, extract_to, description="archive"):
        """Extract zip or tar archive."""
        print(f"Extracting {description}...")
        try:
            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix.lower() in ['.tar', '.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            
            print(f"✓ Extracted {description}")
            return True
        except Exception as e:
            print(f"✗ Failed to extract {description}: {e}")
            return False
    
    def download_sst2(self):
        """Download Stanford Sentiment Treebank (SST-2) using Hugging Face datasets."""
        print("\n" + "="*50)
        print("DOWNLOADING SST-2 (Text Sentiment)")
        print("="*50)
        
        sst2_dir = self.data_dir / "text_sentiment" / "sst2"
        
        if not DATASETS_AVAILABLE:
            print("❌ 'datasets' library not installed!")
            print("Please install it with: pip install datasets")
            print("Falling back to sample data creation...")
            return self._create_sample_sst2(sst2_dir)
        
        # Hugging Face token for authentication
        token = "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"
        
        try:
            print("📥 Downloading SST-2 from Hugging Face...")
            
            # Download different splits with token
            print("  • Loading training data...")
            train_dataset = load_dataset("glue", "sst2", split="train[:2500]", use_auth_token=token)  # 2.5K samples
            
            print("  • Loading validation data...")
            val_dataset = load_dataset("glue", "sst2", split="validation[:400]", use_auth_token=token)   # 400 for validation
            
            print("  • Loading test data...")
            test_dataset = load_dataset("glue", "sst2", split="validation[400:800]", use_auth_token=token)  # 400 for test
            
            # Convert to pandas and save
            datasets_info = [
                (train_dataset, "train"),
                (val_dataset, "dev"), 
                (test_dataset, "test")
            ]
            
            for dataset, split_name in datasets_info:
                # Convert to pandas DataFrame
                df = dataset.to_pandas()
                
                # Rename columns to match our format
                df = df.rename(columns={'sentence': 'text', 'label': 'label'})
                
                # Save as CSV
                output_path = sst2_dir / f"{split_name}.csv"
                df.to_csv(output_path, index=False)
                
                print(f"✓ Saved {split_name}.csv with {len(df)} samples")
                print(f"  Sample: '{df.iloc[0]['text']}' -> {df.iloc[0]['label']}")
            
            print("✅ SST-2 dataset downloaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download SST-2: {e}")
            print("Falling back to sample data creation...")
            return self._create_sample_sst2(sst2_dir)
    
    def _create_sample_sst2(self, sst2_dir):
        """Fallback method to create sample SST-2 data."""
        print("Creating SST-2 sample dataset...")
        
        # Create sample SST-2 data (fallback)
        sample_data = {
            'train': [
                ("I love this movie! It's fantastic.", 1),
                ("This is terrible and boring.", 0),
                ("Great acting and wonderful story.", 1),
                ("Worst movie I've ever seen.", 0),
                ("Pretty good, would recommend.", 1),
                ("Not my cup of tea, quite disappointing.", 0),
                ("Amazing cinematography and direction.", 1),
                ("Could be better, mediocre at best.", 0),
                ("Outstanding performance by all actors.", 1),
                ("Completely waste of time.", 0),
                ("Brilliant screenplay and execution.", 1),
                ("Dull and uninteresting plot.", 0),
                ("Exceptional movie with great message.", 1),
                ("Poor quality and bad acting.", 0),
                ("Highly entertaining and engaging.", 1),
                ("Boring and predictable storyline.", 0),
                ("Masterpiece of modern cinema.", 1),
                ("Disappointing and overrated.", 0),
                ("Wonderful family movie.", 1),
                ("Not worth watching at all.", 0)
            ],
            'dev': [
                ("Good movie with nice plot.", 1),
                ("Bad acting ruins the film.", 0),
                ("Decent entertainment value.", 1),
                ("Completely unwatchable mess.", 0),
                ("Solid storytelling throughout.", 1)
            ],
            'test': [
                ("Excellent direction and cast.", 1),
                ("Terrible script and execution.", 0),
                ("Really enjoyed this film.", 1),
                ("Waste of money and time.", 0)
            ]
        }
        
        # Save as CSV files
        for split, data in sample_data.items():
            df = pd.DataFrame(data, columns=['text', 'label'])
            df.to_csv(sst2_dir / f"{split}.csv", index=False)
            print(f"✓ Created {split}.csv with {len(df)} samples")
        
        print("✓ SST-2 sample dataset ready!")
        return True
    
    def create_emo_db_labels(self):
        """Create labels.csv for existing EMO-DB dataset."""
        print("\n" + "="*50)
        print("CREATING EMO-DB LABELS")
        print("="*50)
        
        emo_db_wav_path = self.data_dir / "audio_emotion" / "emodb" / "wav"
        emo_db_dir = self.data_dir / "audio_emotion" / "emodb"
        
        if not emo_db_wav_path.exists():
            print(f"❌ EMO-DB not found at {emo_db_wav_path}")
            print("Please ensure EMO-DB is downloaded and placed in the correct location.")
            return False
        
        # Get all wav files
        wav_files = list(emo_db_wav_path.glob("*.wav"))
        print(f"📁 Found {len(wav_files)} audio files")
        
        # Create labels from filename convention
        emotion_map = {
            'W': 'anger',
            'L': 'boredom', 
            'E': 'disgust',
            'A': 'fear',
            'F': 'happiness',
            'T': 'sadness',
            'N': 'neutral'
        }
        
        labels_data = []
        for wav_file in wav_files:
            filename = wav_file.name
            # Extract emotion from filename (6th character)
            if len(filename) >= 6:
                emotion_code = filename[5]  # 6th character (0-indexed)
                emotion = emotion_map.get(emotion_code, 'unknown')
                speaker = filename[:2]  # First 2 characters
                
                labels_data.append({
                    'filename': filename,
                    'emotion': emotion,
                    'speaker': speaker,
                    'path': str(wav_file)
                })
        
        # Create labels DataFrame
        labels_df = pd.DataFrame(labels_data)
        labels_csv_path = emo_db_dir / "labels.csv"
        labels_df.to_csv(labels_csv_path, index=False)
        
        print(f"✅ Created {labels_csv_path} with {len(labels_df)} entries")
        print(f"📊 Emotion distribution:")
        emotion_counts = labels_df['emotion'].value_counts()
        for emotion, count in emotion_counts.items():
            print(f"   {emotion}: {count}")
        
        return True
    
    def create_fer2013_labels(self):
        """Create labels.csv for existing FER-2013 dataset."""
        print("\n" + "="*50)
        print("CREATING FER-2013 LABELS")
        print("="*50)
        
        fer_train_path = self.data_dir / "image_emotion" / "fer-2013" / "train"
        fer_test_path = self.data_dir / "image_emotion" / "fer-2013" / "test"
        fer_dir = self.data_dir / "image_emotion" / "fer-2013"
        
        if not fer_train_path.exists():
            print(f"❌ FER-2013 not found at {fer_train_path}")
            print("Please ensure FER-2013 is downloaded and placed in the correct location.")
            return False
        
        labels_data = []
        
        # Process train and test directories
        for split_name, split_path in [("train", fer_train_path), ("test", fer_test_path)]:
            if not split_path.exists():
                print(f"⚠️  {split_name} directory not found: {split_path}")
                continue
                
            print(f"📁 Processing {split_name} directory...")
            
            # Get emotion directories
            emotion_dirs = [d for d in split_path.iterdir() if d.is_dir()]
            
            for emotion_dir in emotion_dirs:
                emotion = emotion_dir.name
                
                # Get all image files
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    image_files.extend(emotion_dir.glob(ext))
                
                print(f"   {emotion}: {len(image_files)} images")
                
                for img_file in image_files:
                    labels_data.append({
                        'filename': img_file.name,
                        'emotion': emotion,
                        'split': split_name,
                        'path': str(img_file.relative_to(fer_dir))
                    })
        
        if not labels_data:
            print("❌ No image files found!")
            return False
        
        # Create labels DataFrame
        labels_df = pd.DataFrame(labels_data)
        labels_csv_path = fer_dir / "labels.csv"
        labels_df.to_csv(labels_csv_path, index=False)
        
        print(f"✅ Created {labels_csv_path} with {len(labels_df)} entries")
        print(f"📊 Emotion distribution:")
        emotion_counts = labels_df['emotion'].value_counts()
        for emotion, count in emotion_counts.items():
            print(f"   {emotion}: {count}")
        
        print(f"📊 Split distribution:")
        split_counts = labels_df['split'].value_counts()
        for split, count in split_counts.items():
            print(f"   {split}: {count}")
        
        return True
    
    def create_sample_multimodal_dataset(self):
        """Create a small combined dataset for testing."""
        print("\n" + "="*50)
        print("CREATING SAMPLE MULTIMODAL DATASET")
        print("="*50)
        
        # Create sample combined dataset
        sample_dir = self.data_dir / "sample_combined"
        sample_dir.mkdir(exist_ok=True)
        
        combined_data = []
        
        # Sample entries combining text, audio emotion labels, and image emotion labels
        samples = [
            {"text": "I'm so happy today!", "text_sentiment": 1, "audio_emotion": "happy", "image_emotion": "happy"},
            {"text": "This is terrible news", "text_sentiment": 0, "audio_emotion": "sad", "image_emotion": "sad"},
            {"text": "Feeling quite angry about this", "text_sentiment": 0, "audio_emotion": "anger", "image_emotion": "angry"},
            {"text": "What a wonderful surprise!", "text_sentiment": 1, "audio_emotion": "surprise", "image_emotion": "surprise"},
            {"text": "I'm scared about the future", "text_sentiment": 0, "audio_emotion": "fear", "image_emotion": "fear"},
            {"text": "Just a normal day", "text_sentiment": 1, "audio_emotion": "neutral", "image_emotion": "neutral"},
            {"text": "This disgusts me completely", "text_sentiment": 0, "audio_emotion": "disgust", "image_emotion": "disgust"},
            {"text": "Amazing performance today!", "text_sentiment": 1, "audio_emotion": "happy", "image_emotion": "happy"}
        ]
        
        for i, sample in enumerate(samples):
            combined_data.append({
                'sample_id': f'sample_{i:03d}',
                'text': sample['text'],
                'text_sentiment': sample['text_sentiment'],
                'audio_file': f'audio_{i:03d}.wav',
                'audio_emotion': sample['audio_emotion'],
                'image_file': f'image_{i:03d}.png',
                'image_emotion': sample['image_emotion'],
                'overall_sentiment': sample['text_sentiment']
            })
        
        # Save combined dataset
        df = pd.DataFrame(combined_data)
        df.to_csv(sample_dir / "combined_dataset.csv", index=False)
        
        print(f"✓ Created sample combined dataset with {len(df)} entries")
        return True

def main():
    print("🎯 Multi-Modal Dataset Setup & Verification")
    print("="*50)
    print("This script will:")
    print("1. Download SST-2 (Text Sentiment) - 2,500 real samples")
    print("2. Verify and create labels for existing EMO-DB")
    print("3. Verify and create labels for existing FER-2013")
    print("4. Create sample combined dataset")
    
    if not DATASETS_AVAILABLE:
        print("\n⚠️  OPTIONAL: Install 'datasets' for real SST-2 data:")
        print("   pip install datasets")
        print("   (Will use sample data if not installed)")
    
    downloader = DatasetDownloader()
    
    # Process datasets
    success_count = 0
    
    if downloader.download_sst2():
        success_count += 1
    
    if downloader.create_emo_db_labels():
        success_count += 1
    
    if downloader.create_fer2013_labels():
        success_count += 1
    
    if downloader.create_sample_multimodal_dataset():
        success_count += 1
    
    print(f"\n🎉 Dataset setup completed!")
    print(f"✓ {success_count}/4 tasks completed successfully")
    print("\n📋 What's ready:")
    print("• Text: SST-2 sentiment data (downloaded or sample)")
    print("• Audio: EMO-DB emotion labels (labels.csv created)")
    print("• Image: FER-2013 emotion labels (labels.csv created)")
    print("• Combined: Sample multimodal dataset")
    print("\n🚀 Ready to create data loaders and train!")

if __name__ == "__main__":
    main()
