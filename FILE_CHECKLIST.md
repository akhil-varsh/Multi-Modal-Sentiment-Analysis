# 🎯 Quick Reference: Files You Need to Create

Based on your `train.csv` and `val.csv`, here are all the audio and image files you need to prepare:

## 📁 Training Set (36 files total)

### Audio Files (in `data/my_audio/`)
**Positive (12 files):**
- positive_01.wav to positive_12.wav

**Negative (12 files):**
- negative_01.wav to negative_12.wav

**Neutral (12 files):**
- neutral_01.wav to neutral_12.wav

### Image Files (in `data/my_images/`)
**Happy faces (12 files):**
- happy_face_01.jpg to happy_face_12.jpg

**Sad faces (12 files):**
- sad_face_01.jpg to sad_face_12.jpg

**Neutral faces (12 files):**
- neutral_face_01.jpg to neutral_face_12.jpg

---

## 📁 Validation Set (9 files total)

### Audio Files (in `data/my_audio/`)
- positive_val_01.wav to positive_val_03.wav
- negative_val_01.wav to negative_val_03.wav
- neutral_val_01.wav to neutral_val_03.wav

### Image Files (in `data/my_images/`)
- happy_face_val_01.jpg to happy_face_val_03.jpg
- sad_face_val_01.jpg to sad_face_val_03.jpg
- neutral_face_val_01.jpg to neutral_face_val_03.jpg

---

## 🎙️ Audio Recording Tips

1. **Use Text from CSV**: Read the corresponding text from `train.csv`/`val.csv` in the emotion that matches the label
2. **Tone Matching**:
   - Positive: Happy, excited, enthusiastic tone
   - Negative: Angry, sad, frustrated tone
   - Neutral: Calm, monotone, matter-of-fact
3. **Recording Settings**: 
   - Sample rate: 16kHz (or record at higher and convert)
   - Format: WAV (mono preferred)
   - Duration: ~3-5 seconds

**Free Recording Tools:**
- Audacity (desktop)
- Voice Recorder (Windows built-in)
- Online: https://online-voice-recorder.com/

---

## 📸 Image Sourcing Tips

1. **Facial Expressions**:
   - Positive: Smiling, laughing faces
   - Negative: Frowning, crying, angry faces
   - Neutral: Relaxed, expressionless faces

2. **Sources**:
   - Take selfies with appropriate expressions
   - Use royalty-free image sites:
     - Unsplash: https://unsplash.com/s/photos/facial-expression
     - Pexels: https://www.pexels.com/search/emotion/
     - Pixabay: https://pixabay.com/images/search/face/
   - Use generated images (with AI tools)

3. **Image Format**:
   - JPG or PNG
   - Any size (system will resize to 224x224)

---

## ⚡ Quick Start Option: Using Placeholders

If you want to test the system first before creating real data:

```python
# Run this to create dummy audio/image files
python create_dummy_files.py
```

(I'll create this script for you next!)

---

## ✅ Verification

Once you've added your files, verify with:
```bash
# Check audio files
dir data\my_audio

# Check image files  
dir data\my_images

# Should see 45 audio files and 45 image files total
```

Then run training:
```bash
python train.py
```
