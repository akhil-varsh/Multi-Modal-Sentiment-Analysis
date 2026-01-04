# 🛠️ Custom Dataset Guide

This guide explains how to prepare your own data for the Multi-Modal Sentiment Analysis system.

---

## 📂 Option 1: Metadata File (Recommended)

This is the most standard approach. You create a CSV file that lists the paths to your text, audio, and image files.

### 1. Folder Structure
Organize your raw files however you like:
```
data/
├── train.csv
├── val.csv
├── my_audio/
│   ├── sample1.wav
│   └── ...
├── my_images/
│   ├── face1.jpg
│   └── ...
```

### 2. CSV Format
Create a `train.csv` (and `val.csv`) in the `data/` folder with these columns:
- `text`: The text content.
- `audio_path`: Relative path to the audio file.
- `image_path`: Relative path to the image file.
- `label`: Integer representing sentiment (**0: Negative, 1: Neutral, 2: Positive**).

**Example `train.csv`:**
```csv
text,audio_path,image_path,label
"I am so happy today!",my_audio/sample1.wav,my_images/face1.jpg,2
"This is very disappointing.",my_audio/sad.wav,my_images/face2.jpg,0
"I am waiting for the bus.",my_audio/ambient.wav,my_images/street.jpg,1
```

---

## 📁 Option 2: Folder-Based Structure (Quick Setup)

If you don't want to create a CSV, the system can automatically pair files based on their **filenames**.

### 1. Folder Structure
Place your files in these exact subdirectories under `data/`:
```
data/
├── text_sentiment/
│   ├── sample_01.txt
│   └── sample_02.txt
├── audio_emotion/
│   ├── sample_01.wav
│   └── sample_02.wav
├── image_emotion/
│   ├── sample_01.jpg
│   └── sample_02.png
```

### 2. How it works
The system scans `text_sentiment/` first. For every `.txt` file (e.g., `sample_01.txt`), it looks for a file with the **same name** in the audio and image folders (e.g., `sample_01.wav` and `sample_01.jpg`).

> [!NOTE]
> Labels for this method default to **1 (Neutral)**. Use Option 1 if you need specific labels for training.

---

## 📏 Technical Requirements

To get the best results, ensure your data meets these specs:

| Modality | Requirement |
|----------|-------------|
| **Text** | Clean UTF-8 encoded plain text. |
| **Audio** | **16kHz sampling rate** mono (WAV or MP3). System will resample if needed. |
| **Image**| RGB format (JPG/PNG). System will resize to 224x224. |
| **Labels**| **0**: Negative \| **1**: Neutral \| **2**: Positive |

---

## 🚀 How to Use Your Dataset

1.  **Clear existing data**: Delete any temporary dummy files in `data/`.
2.  **Add your files**: Use one of the options above.
3.  **Run Training**:
    ```bash
    python train.py
    ```
    The script will automatically detect your local files and start training.

4.  **Verify**: Keep an eye on the console logs. It will report how many samples were found.
    ```
    Scanning data for samples...
    - Training samples: 500
    - Validation samples: 50
    ```

---

## 🎨 Visualization
Once trained, launch the app:
```bash
streamlit run app.py
```
Upload your own files in the app to see how the model generalizes to your custom data!
