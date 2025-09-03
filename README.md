---
title: Multimodal Sentiment Analysis
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.48.1"
app_file: app.py
pinned: false
---

# Multimodal Sentiment Analysis

A comprehensive Streamlit application that combines three different sentiment analysis models: text, audio, and vision-based sentiment analysis. The project demonstrates how to integrate multiple AI models for comprehensive sentiment understanding across different modalities.

![Demo](https://user-images.githubusercontent.com/.../video.mp4)

## What is it?

This project implements a **fused sentiment analysis system** that combines predictions from three independent models:

### 1. Text Sentiment Analysis

- **Model**: TextBlob NLP library
- **Capability**: Analyzes text input for positive, negative, or neutral sentiment
- **Status**: ✅ Fully integrated and ready to use

### 2. Audio Sentiment Analysis

- **Model**: Fine-tuned Wav2Vec2-base model
- **Training Data**: RAVDESS + CREMA-D emotional speech datasets
- **Capability**: Analyzes audio files and microphone recordings for sentiment
- **Features**:
  - File upload support (WAV, MP3, M4A, FLAC)
  - Direct microphone recording (max 5 seconds)
  - Automatic preprocessing (16kHz sampling, 5s max duration)
- **Status**: ✅ Fully integrated and ready to use

### 3. Vision Sentiment Analysis

- **Model**: Fine-tuned ResNet-50 model
- **Training Data**: FER2013 facial expression dataset
- **Capability**: Analyzes images for facial expression-based sentiment
- **Features**:
  - File upload support (PNG, JPG, JPEG, BMP, TIFF)
  - Camera capture functionality
  - Automatic face detection and preprocessing
  - Grayscale conversion and 224x224 resize
- **Status**: ✅ Fully integrated and ready to use

### 4. Fused Model

- **Approach**: Combines predictions from all three models
- **Capability**: Provides comprehensive sentiment analysis across modalities
- **Status**: ✅ Fully integrated and ready to use

### 5. 🎬 Max Fusion

- **Approach**: Video-based comprehensive sentiment analysis
- **Capability**: Analyzes 5-second videos by extracting frames, audio, and transcribing speech
- **Features**:
  - Video recording or file upload (MP4, AVI, MOV, MKV, WMV, FLV)
  - Automatic frame extraction for vision analysis
  - Audio extraction for vocal sentiment analysis
  - Speech-to-text transcription for text sentiment analysis
  - Combined results from all three modalities
- **Status**: ✅ Fully integrated and ready to use

## Project Structure

```
sentiment-fused/
├── app.py                          # Main Streamlit application
├── simple_model_manager.py         # Model management and Google Drive integration
├── requirements.txt                # Python dependencies
├── pyproject.toml                 # Project configuration
├── Dockerfile                     # Container deployment
├── notebooks/                     # Development notebooks
│   ├── audio_sentiment_analysis.ipynb    # Audio model development
│   └── vision_sentiment_analysis.ipynb   # Vision model development
├── model_weights/                 # Model storage directory (downloaded .pth files)
└── src/                           # Source code package
    ├── __init__.py               # Package initialization
    ├── config/                   # Configuration settings
    ├── models/                   # Model logic and inference code
    ├── utils/                    # Utility functions and preprocessing
    └── ui/                       # User interface components
```

## Key Features

- **Real-time Analysis**: Instant sentiment predictions with confidence scores
- **Smart Preprocessing**: Automatic file format handling and preprocessing
- **Multi-Page Interface**: Clean navigation between different sentiment analysis modes
- **Model Management**: Automatic model downloading from Google Drive
- **File Support**: Multiple audio and image format support
- **Camera & Microphone**: Direct input capture capabilities

## Prerequisites

- Python 3.9 or higher
- 4GB+ RAM (for model loading)
- Internet connection (for initial model download)

## Installation

1. **Clone the repository**:

   ```bash
   git clone <your-repo-url>
   cd sentiment-fused
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root with:
   ```env
   VISION_MODEL_DRIVE_ID=your_google_drive_vision_model_file_id_here
   AUDIO_MODEL_DRIVE_ID=your_google_drive_audio_model_file_id_here
   VISION_MODEL_FILENAME=resnet50_model.pth
   AUDIO_MODEL_FILENAME=wav2vec2_model.pth
   ```

## Running Locally

1. **Start the Streamlit application**:

   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. **Navigate between pages** using the sidebar:
   - 🏠 **Home**: Overview and welcome page
   - 📝 **Text Sentiment**: Analyze text with TextBlob
   - 🎵 **Audio Sentiment**: Analyze audio files or record with microphone
   - 🖼️ **Vision Sentiment**: Analyze images or capture with camera
   - 🔗 **Fused Model**: Combine all three models
   - 🎬 **Max Fusion**: Video-based comprehensive analysis

## Model Development

The project includes Jupyter notebooks that document the development process:

### Audio Model (`notebooks/audio_sentiment_analysis.ipynb`)

- Wav2Vec2-base fine-tuning on RAVDESS + CREMA-D datasets
- Emotion-to-sentiment mapping (happy/surprised → positive, sad/angry/fearful/disgust → negative, neutral/calm → neutral)
- Audio preprocessing pipeline (16kHz sampling, 5s max duration)

### Vision Model (`notebooks/vision_sentiment_analysis.ipynb`)

- ResNet-50 fine-tuning on FER2013 dataset
- Emotion-to-sentiment mapping (happy/surprise → positive, angry/disgust/fear/sad → negative, neutral → neutral)
- Image preprocessing pipeline (face detection, grayscale conversion, 224x224 resize)

## Technical Implementation

### Model Management

- `SimpleModelManager` class handles model downloading from Google Drive
- Automatic model caching and version management
- Environment variable configuration for model URLs

### Preprocessing Pipelines

- **Audio**: Automatic resampling, duration limiting, feature extraction
- **Vision**: Face detection, cropping, grayscale conversion, normalization
- **Text**: Direct TextBlob processing

### Streamlit Integration

- Multi-page application with sidebar navigation
- File upload widgets with format validation
- Real-time camera and microphone input
- Custom CSS styling for modern UI

## Deployment

### Docker Deployment

```bash
# Build the container
docker build -t sentiment-fused .

# Run the container
doc

Uploading multimodal-sentiment-analysis-video-demo.mp4…

ker run -p 7860:7860 sentiment-fused
```

The application will be available at `http://localhost:7860`

### Local Development

```bash
# Run with custom port
streamlit run app.py --server.port 8502

# Run with custom address
streamlit run app.py --server.address 0.0.0.0
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:

   - Ensure environment variables are set correctly
   - Check internet connection for model downloads
   - Verify sufficient RAM (4GB+ recommended)

2. **Dependency Issues**:

   - Use virtual environment to avoid conflicts
   - Install PyTorch with CUDA support if using GPU
   - Ensure OpenCV is properly installed for face detection

3. **Performance Issues**:
   - Large audio/image files may cause memory issues
   - Consider file size limits for better performance
   - GPU acceleration available for PyTorch models

### Model Testing

```bash
# Test vision model
python -c "from simple_model_manager import SimpleModelManager; m = SimpleModelManager(); print('Vision model:', m.load_vision_model()[0] is not None)"

# Test audio model
python -c "from simple_model_manager import SimpleModelManager; m = SimpleModelManager(); print('Audio model:', m.load_audio_model()[0] is not None)"
```

## Dependencies

Key libraries used:

- **Streamlit**: Web application framework
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face model library
- **OpenCV**: Computer vision and face detection
- **Librosa**: Audio processing
- **TextBlob**: Natural language processing
- **Gdown**: Google Drive file downloader
- **MoviePy**: Video processing and audio extraction
- **SpeechRecognition**: Audio transcription

## What This Project Demonstrates

1. **Multimodal AI Integration**: Combining text, audio, and vision models
2. **Model Management**: Automated downloading and caching of pre-trained models
3. **Real-time Processing**: Live audio recording and camera capture
4. **Smart Preprocessing**: Automatic format conversion and optimization
5. **Modern Web UI**: Professional Streamlit application with custom styling
6. **Production Ready**: Docker containerization and deployment
7. **Video Analysis**: Comprehensive video processing with multi-modal extraction
8. **Speech Recognition**: Audio-to-text transcription for enhanced analysis
9. **Modular Architecture**: Clean, maintainable code structure with separated concerns
10. **Professional Code Organization**: Proper Python packaging with config, models, utils, and UI modules

## Recent Improvements

The project has been refactored from a monolithic structure to a clean, modular architecture:

- **Modular Design**: Separated into logical modules (`src/config/`, `src/models/`, `src/utils/`, `src/ui/`)
- **Centralized Configuration**: All settings consolidated in `src/config/settings.py`
- **Clean Separation**: Model logic, preprocessing, and UI components are now in dedicated modules
- **Better Maintainability**: Easier to modify, test, and extend individual components
- **Professional Structure**: Follows Python packaging best practices

This project serves as a comprehensive example of building production-ready multimodal AI applications with modern Python tools and frameworks.
