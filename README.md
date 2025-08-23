# Sentiment Analysis Testing Ground

A comprehensive multi-page Streamlit application for testing three independent sentiment analysis models: text, audio, and vision-based sentiment analysis.

## 🚀 Features

- **Multi-Page Interface**: Clean navigation with dedicated pages for each model
- **Text Sentiment Analysis**: ✅ **READY TO USE** - TextBlob NLP model integrated
- **Audio Sentiment Analysis**: ✅ **READY TO USE** - Fine-tuned Wav2Vec2 model integrated
  - 📁 **File Upload**: Support for WAV, MP3, M4A, FLAC files
  - 🎙️ **Audio Recording**: Direct microphone recording (max 5 seconds)
  - 🔄 **Smart Preprocessing**: Automatic 16kHz sampling, 5s max duration (CREMA-D + RAVDESS format)
- **Vision Sentiment Analysis**: ✅ **READY TO USE** - Fine-tuned ResNet-50 model integrated
  - 📁 **File Upload**: Support for PNG, JPG, JPEG, BMP, TIFF files
  - 📷 **Camera Capture**: Take photos directly with your camera
  - 🔄 **Smart Preprocessing**: Automatic face detection, tight face crop (0% padding), grayscale conversion, 224x224 resize
- **Fused Model**: Combine predictions from all three models
- **Modern UI**: Beautiful, responsive interface with custom styling
- **File Support**: Multiple audio and image format support

## 📋 Requirements

- Python 3.9 or higher
- Streamlit 1.28.0 or higher
- PyTorch 1.13.0 or higher
- Additional dependencies listed in `requirements.txt`

## 🛠️ Installation

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

## 🚀 Usage

1. **Start the Streamlit application**:

   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. **Navigate between pages** using the sidebar:
   - 🏠 **Home**: Overview and welcome page
   - 📝 **Text Sentiment**: ✅ **Ready to use** - Analyze text with TextBlob
   - 🎵 **Audio Sentiment**: ✅ **Ready to use** - Analyze audio with Wav2Vec2 - 📁 Upload audio files or 🎙️ record directly with microphone using `st.audio_input`
   - 🖼️ **Vision Sentiment**: ✅ **Ready to use** - Analyze images with ResNet-50
     - 📁 Upload image files or 📷 take photos with camera
   - 🔗 **Fused Model**: Combine all three models

## 🧪 Testing the Models

Before running the full app, you can test if the models load correctly:

### Vision Model Test

```bash
python test_vision_model.py
```

### Audio Model Test

```bash
python test_audio_model.py
```

These will verify that:

- The model files exist
- PyTorch can load the architectures
- The trained weights can be loaded
- Inference runs without errors

### 🔍 Troubleshooting Model Issues

If you encounter tensor size mismatch errors, run the diagnostic scripts:

```bash
python check_model.py          # For vision model
python test_audio_model.py     # For audio model
```

These will examine your model files and identify:

- The actual number of output classes
- Whether the architectures match expected models
- Any compatibility issues

**Common Issues:**

- **Tensor size mismatch**: Models might have been trained with different numbers of classes
- **Architecture mismatch**: Models might not match expected architectures
- **Weight loading errors**: Corrupted or incompatible model files
- **Library dependencies**: Missing transformers, librosa, or other required libraries

## 📁 Project Structure

```
sentiment-fused/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── test_vision_model.py           # Vision model test script
├── test_audio_model.py            # Audio model test script
├── main.py                        # Original main file
├── pyproject.toml                 # Project configuration
└── models/                        # Model files and notebooks
    ├── audio_sentiment_analysis.ipynb
    ├── vision_sentiment_analysis.ipynb
    ├── wav2vec2_model.pth        # ✅ Fine-tuned Wav2Vec2 model (READY)
    └── resnet50_model.pth        # ✅ Fine-tuned ResNet-50 model (READY)
```

## 🔧 Model Integration Status

### ✅ Text Sentiment Model - **READY TO USE**

- **Model**: TextBlob (Natural Language Processing)
- **Features**: Sentiment classification (Positive/Negative/Neutral) with confidence scores
- **Input**: Any text input
- **Analysis**: Real-time NLP sentiment analysis
- **Status**: Fully integrated and tested

### ✅ Vision Sentiment Model - **READY TO USE**

- **Model**: ResNet-50 fine-tuned on FER2013 dataset
- **Training Dataset**:
  - 🖼️ **FER2013**: Facial Expression Recognition 2013 dataset
  - 🎯 **Classes**: 7 emotions mapped to 3 sentiments (Negative, Neutral, Positive)
  - 🏗️ **Architecture**: ResNet-50 with ImageNet weights, fine-tuned for sentiment
- **Classes**: 3 sentiment classes (Negative, Neutral, Positive)
- **Input**: Images (PNG, JPG, JPEG, BMP, TIFF)
  - **Preprocessing**:
    - 🔍 **Face Detection**: Automatic face detection using OpenCV
    - 🎨 **Grayscale Conversion**: Convert to grayscale and replicate to 3 channels
    - 📏 **Face Cropping**: Crop to face region with 0% padding (tightest crop)
    - 📐 **Resize**: Scale to 224x224 pixels (FER2013 format)
    - 🎯 **Transforms**: Resize(224) → CenterCrop(224) → ToTensor → ImageNet Normalization
    - 📊 **Format**: 224x224 RGB with ImageNet mean/std normalization
- **Status**: Fully integrated and tested

### ✅ Audio Sentiment Model - **READY TO USE**

- **Model**: Wav2Vec2-base fine-tuned on RAVDESS + CREMA-D datasets
- **Training Datasets**:
  - 🎵 **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song
  - 🎵 **CREMA-D**: Crowd-sourced Emotional Multimodal Actors Dataset
- **Classes**: 3 sentiment classes (Negative, Neutral, Positive)
- **Input**:
  - 📁 **File Upload**: Audio files (WAV, MP3, M4A, FLAC)
  - 🎙️ **Direct Recording**: Microphone input using `st.audio_input`
  - **Preprocessing**:
    - 🔄 **Sampling Rate**: 16kHz (matching CREMA-D + RAVDESS training)
    - ⏱️ **Duration**: Max 5 seconds (matching training max_duration_s=5.0)
    - 🎵 **Feature Extraction**: AutoFeatureExtractor with truncation and padding
    - 📊 **Format**: Automatic resampling, max_length=int(5.0 \* 16000)
- **Status**: Fully integrated and tested

### 🔗 Fused Model - **FULLY READY**

The fused model now uses all three integrated models: text (TextBlob), audio (Wav2Vec2), and vision (ResNet-50).

## 📊 Supported File Formats

### Audio Files

- WAV (.wav)
- MP3 (.mp3)
- M4A (.m4a)
- FLAC (.flac)

### Image Files

- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)
- TIFF (.tiff)

## 🎨 Customization

The application includes custom CSS styling that can be modified in the `app.py` file. Key styling classes:

- `.main-header`: Main page headers
- `.model-card`: Information cards
- `.result-box`: Result display boxes
- `.upload-section`: File upload areas

## 🔍 Troubleshooting

### Common Issues

1. **Port already in use**: Change the port with `streamlit run app.py --server.port 8502`

2. **Vision model loading errors**:

   - Ensure `models/resnet50_model.pth` exists
   - Run `python test_vision_model.py` to diagnose issues
   - Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`

3. **Memory issues**: Large audio/image files may require more memory. Consider file size limits

4. **OpenCV issues**: If face detection fails, ensure `opencv-python` is installed:

   ```bash
   pip install opencv-python
   ```

5. **Dependency conflicts**: Use a virtual environment to avoid package conflicts

### Performance Tips

- Use appropriate file sizes for audio and images
- Consider implementing caching for model predictions
- Use GPU acceleration if available for PyTorch models
- The vision model automatically uses GPU if available

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Streamlit team for the amazing web framework
- PyTorch community for deep learning tools
- Hugging Face for transformer models
- All contributors to the open-source libraries used

## 📞 Support

For questions or issues:

1. Check the troubleshooting section above
2. Run `python test_vision_model.py` for vision model issues
3. Review the model integration examples
4. Open an issue on the repository
5. Contact the development team

---

**Happy Sentiment Analysis! 🧠✨**

**Note**: All **THREE MODELS** are now fully integrated and ready to use! 🎉
