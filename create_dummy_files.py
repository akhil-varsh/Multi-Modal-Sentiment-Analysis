"""
Create dummy placeholder audio and image files for testing
This allows you to run the training pipeline immediately while you prepare real data
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import wave
import struct
from pathlib import Path

def create_dummy_audio(filename, duration=3, sample_rate=16000):
    """Create a simple sine wave audio file"""
    output_path = Path('data/my_audio') / filename
    
    # Generate sine wave
    frequency = 440  # A4 note
    num_samples = int(duration * sample_rate)
    
    # Create waveform
    waveform = []
    for i in range(num_samples):
        value = int(32767 * 0.3 * np.sin(2 * np.pi * frequency * i / sample_rate))
        waveform.append(value)
    
    # Write WAV file
    with wave.open(str(output_path), 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack('h' * len(waveform), *waveform))
    
    return output_path

def create_dummy_image(filename, emotion="neutral"):
    """Create a simple colored placeholder image with text"""
    output_path = Path('data/my_images') / filename
    
    # Color scheme based on emotion
    colors = {
        'happy': (255, 235, 59),    # Yellow
        'sad': (33, 150, 243),      # Blue
        'neutral': (158, 158, 158)  # Gray
    }
    
    # Determine emotion from filename
    if 'happy' in filename:
        emotion = 'happy'
        emoji = '😊'
    elif 'sad' in filename:
        emotion = 'sad'
        emoji = '😞'
    else:
        emotion = 'neutral'
        emoji = '😐'
    
    # Create image
    img = Image.new('RGB', (224, 224), color=colors[emotion])
    draw = ImageDraw.Draw(img)
    
    # Add text (emoji as placeholder)
    try:
        # Try to use a font (may not work on all systems)
        font = ImageFont.truetype("arial.ttf", 80)
    except:
        font = ImageFont.load_default()
    
    # Draw emoji in center
    bbox = draw.textbbox((0, 0), emoji, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((224 - text_width) // 2, (224 - text_height) // 2)
    draw.text(position, emoji, fill=(255, 255, 255), font=font)
    
    # Save
    img.save(output_path)
    return output_path

def main():
    print("🎬 Creating dummy audio and image files...")
    print()
    
    # Audio files to create
    audio_files = [
        # Training
        *[f'positive_{i:02d}.wav' for i in range(1, 13)],
        *[f'negative_{i:02d}.wav' for i in range(1, 13)],
        *[f'neutral_{i:02d}.wav' for i in range(1, 13)],
        # Validation
        *[f'positive_val_{i:02d}.wav' for i in range(1, 4)],
        *[f'negative_val_{i:02d}.wav' for i in range(1, 4)],
        *[f'neutral_val_{i:02d}.wav' for i in range(1, 4)]
    ]
    
    # Image files to create
    image_files = [
        # Training
        *[f'happy_face_{i:02d}.jpg' for i in range(1, 13)],
        *[f'sad_face_{i:02d}.jpg' for i in range(1, 13)],
        *[f'neutral_face_{i:02d}.jpg' for i in range(1, 13)],
        # Validation
        *[f'happy_face_val_{i:02d}.jpg' for i in range(1, 4)],
        *[f'sad_face_val_{i:02d}.jpg' for i in range(1, 4)],
        *[f'neutral_face_val_{i:02d}.jpg' for i in range(1, 4)]
    ]
    
    # Create audio files
    print("🔊 Creating audio files...")
    for audio_file in audio_files:
        create_dummy_audio(audio_file)
    print(f"   ✅ Created {len(audio_files)} audio files")
    
    # Create image files
    print("🖼️  Creating image files...")
    for image_file in image_files:
        create_dummy_image(image_file)
    print(f"   ✅ Created {len(image_files)} image files")
    
    print()
    print("✅ All dummy files created!")
    print()
    print("📂 Locations:")
    print(f"   Audio: data/my_audio/ ({len(audio_files)} files)")
    print(f"   Images: data/my_images/ ({len(image_files)} files)")
    print()
    print("🚀 You can now run training:")
    print("   python train.py")
    print()
    print("💡 Replace these dummy files with real recordings/images when ready!")

if __name__ == "__main__":
    main()
