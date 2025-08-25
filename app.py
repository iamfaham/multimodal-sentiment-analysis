import streamlit as st
import pandas as pd
from PIL import Image
import io
import numpy as np
import tempfile
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.nn.functional as F

# Import the Google Drive model manager
from simple_model_manager import SimpleModelManager

# Page configuration
st.set_page_config(
    page_title="Multimodal Sentiment Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .result-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Initialize the Google Drive model manager
@st.cache_resource
def get_model_manager():
    """Get the Google Drive model manager instance"""
    try:
        manager = SimpleModelManager()
        return manager
    except Exception as e:
        st.error(f"Failed to initialize model manager: {e}")
        return None


# Global variables for models
@st.cache_resource
def load_vision_model():
    """Load the pre-trained ResNet-50 vision sentiment model from Google Drive"""
    try:
        manager = get_model_manager()
        if manager is None:
            st.error("Model manager not available")
            return None, None, None

        # Load the model using the Google Drive manager
        model, device, num_classes = manager.load_vision_model()

        if model is None:
            st.error("Failed to load vision model from Google Drive")
            return None, None, None

        st.success(f"Vision model loaded successfully with {num_classes} classes!")
        return model, device, num_classes
    except Exception as e:
        st.error(f"Error loading vision model: {str(e)}")
        return None, None, None


@st.cache_data
def get_vision_transforms():
    """Get the image transforms used during FER2013 training"""
    return transforms.Compose(
        [
            transforms.Resize(224),  # Match training: transforms.Resize(224)
            transforms.CenterCrop(224),  # Match training: transforms.CenterCrop(224)
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet normalization
        ]
    )


def detect_and_preprocess_face(image, crop_tightness=0.05):
    """
    Detect face in image, crop to face region, convert to grayscale, and resize to 224x224
    to match FER2013 dataset format (grayscale converted to 3-channel RGB)

    Args:
        image: Input image (PIL Image or numpy array)
        crop_tightness: Padding around face (0.0 = no padding, 0.3 = 30% padding)
    """
    try:
        import cv2
        import numpy as np

        # Convert PIL image to OpenCV format
        if isinstance(image, Image.Image):
            # Convert PIL to numpy array
            img_array = np.array(image)
            # Convert RGB to BGR for OpenCV
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_array = image

        # Load face detection cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Convert to grayscale for face detection (detection works better on grayscale)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            st.warning("No face detected in the image. Using center crop instead.")
            # Fallback: center crop and resize
            if isinstance(image, Image.Image):
                # Convert to RGB first
                rgb_pil = image.convert("RGB")
                # Center crop to square
                width, height = rgb_pil.size
                size = min(width, height)
                left = (width - size) // 2
                top = (height - size) // 2
                right = left + size
                bottom = top + size
                cropped = rgb_pil.crop((left, top, right, bottom))
                # Resize to 224x224 (matching FER2013 training: transforms.Resize(224))
                resized = cropped.resize((224, 224), Image.Resampling.LANCZOS)

                # Convert to grayscale and then to 3-channel RGB
                gray_pil = resized.convert("L")
                # Convert back to RGB (this replicates grayscale values to all 3 channels)
                gray_rgb_pil = gray_pil.convert("RGB")
                return gray_rgb_pil
            else:
                return None

        # Get the largest face (assuming it's the main subject)
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

        # Add padding around the face based on user preference
        padding_x = int(w * crop_tightness)
        padding_y = int(h * crop_tightness)

        # Ensure we don't go out of bounds
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(img_array.shape[1], x + w + padding_x)
        y2 = min(img_array.shape[0], y + h + padding_y)

        # Crop to face region
        face_crop = img_array[y1:y2, x1:x2]

        # Convert BGR to RGB first
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # Convert to grayscale
        face_gray = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2GRAY)

        # Resize to 224x224 (matching FER2013 training: transforms.Resize(224))
        face_resized = cv2.resize(face_gray, (224, 224), interpolation=cv2.INTER_AREA)

        # Convert grayscale to 3-channel RGB (replicate grayscale values)
        face_rgb_3channel = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)

        # Convert back to PIL Image
        face_pil = Image.fromarray(face_rgb_3channel)

        return face_pil

    except ImportError:
        st.error(
            "OpenCV not installed. Please install it with: pip install opencv-python"
        )
        st.info("Falling back to basic preprocessing...")
        # Fallback: basic grayscale conversion and resize
        if isinstance(image, Image.Image):
            rgb_pil = image.convert("RGB")
            resized = rgb_pil.resize((48, 48), Image.Resampling.LANCZOS)
            # Convert to grayscale and then to 3-channel RGB
            gray_pil = resized.convert("L")
            gray_rgb_pil = gray_pil.convert("RGB")
            return gray_rgb_pil
        return None
    except Exception as e:
        st.error(f"Error in face detection: {str(e)}")
        st.info("Falling back to basic preprocessing...")
        # Fallback: basic grayscale conversion and resize
        if isinstance(image, Image.Image):
            rgb_pil = image.convert("RGB")
            resized = rgb_pil.resize((48, 48), Image.Resampling.LANCZOS)
            # Convert to grayscale and then to 3-channel RGB
            gray_pil = resized.convert("L")
            gray_rgb_pil = gray_pil.convert("RGB")
            return gray_rgb_pil
        return None


def get_sentiment_mapping(num_classes):
    """Get the sentiment mapping based on number of classes"""
    if num_classes == 3:
        return {0: "Negative", 1: "Neutral", 2: "Positive"}
    elif num_classes == 4:
        # Common 4-class emotion mapping
        return {0: "Angry", 1: "Sad", 2: "Happy", 3: "Neutral"}
    elif num_classes == 7:
        # FER2013 7-class emotion mapping
        return {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Sad",
            5: "Surprise",
            6: "Neutral",
        }
    else:
        # Generic mapping for unknown number of classes
        return {i: f"Class_{i}" for i in range(num_classes)}


# Placeholder functions for model predictions
def predict_text_sentiment(text):
    """
    Analyze text sentiment using TextBlob
    """
    if not text or text.strip() == "":
        return "No text provided", 0.0

    try:
        from textblob import TextBlob

        # Create TextBlob object
        blob = TextBlob(text)

        # Get polarity (-1 to 1, where -1 is very negative, 1 is very positive)
        polarity = blob.sentiment.polarity

        # Get subjectivity (0 to 1, where 0 is very objective, 1 is very subjective)
        subjectivity = blob.sentiment.subjectivity

        # Convert polarity to sentiment categories
        if polarity > 0.1:
            sentiment = "Positive"
            confidence = min(0.95, 0.6 + abs(polarity) * 0.3)
        elif polarity < -0.1:
            sentiment = "Negative"
            confidence = min(0.95, 0.6 + abs(polarity) * 0.3)
        else:
            sentiment = "Neutral"
            confidence = 0.7 - abs(polarity) * 0.2

        # Round confidence to 2 decimal places
        confidence = round(confidence, 2)

        return sentiment, confidence

    except ImportError:
        st.error("TextBlob not installed. Please install it with: pip install textblob")
        return "TextBlob not available", 0.0
    except Exception as e:
        st.error(f"Error in text sentiment analysis: {str(e)}")
        return "Error occurred", 0.0


@st.cache_resource
def load_audio_model():
    """Load the pre-trained Wav2Vec2 audio sentiment model from Google Drive"""
    try:
        manager = get_model_manager()
        if manager is None:
            st.error("Model manager not available")
            return None, None, None, None

        # Load the model using the Google Drive manager
        model, device = manager.load_audio_model()

        if model is None:
            st.error("Failed to load audio model from Google Drive")
            return None, None, None, None

        # For Wav2Vec2 models, we need to determine the number of classes
        # This is typically available in the model configuration
        try:
            num_classes = model.config.num_labels
        except:
            # Fallback: try to infer from the model
            try:
                num_classes = model.classifier.out_features
            except:
                num_classes = 3  # Default assumption

        # Load feature extractor
        from transformers import AutoFeatureExtractor

        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base"
        )

        st.success(f"Audio model loaded successfully with {num_classes} classes!")
        return model, device, num_classes, feature_extractor
    except Exception as e:
        st.error(f"Error loading audio model: {str(e)}")
        return None, None, None, None


def predict_audio_sentiment(audio_bytes):
    """
    Analyze audio sentiment using fine-tuned Wav2Vec2 model
    Preprocessing matches CREMA-D + RAVDESS training specifications:
    - Target sampling rate: 16kHz
    - Max duration: 5.0 seconds
    - Feature extraction: AutoFeatureExtractor with max_length, truncation, padding
    """
    if audio_bytes is None:
        return "No audio provided", 0.0

    try:
        # Load model if not already loaded
        model, device, num_classes, feature_extractor = load_audio_model()
        if model is None:
            return "Model not loaded", 0.0

        # Load and preprocess audio
        import librosa
        import io
        import tempfile

        # Save audio bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name

        try:
            # Load audio with librosa
            audio, sr = librosa.load(tmp_file_path, sr=None)

            # Resample to 16kHz if needed
            if sr != 16000:
                audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16000)

            # Preprocess with feature extractor (matching CREMA-D + RAVDESS training exactly)
            # From training: max_length=int(max_duration_s * TARGET_SAMPLING_RATE) = 5.0 * 16000
            inputs = feature_extractor(
                audio,
                sampling_rate=16000,
                max_length=int(5.0 * 16000),  # 5 seconds max (matching training)
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            # Move to device
            input_values = inputs.input_values.to(device)

            # Run inference
            with torch.no_grad():
                outputs = model(input_values)
                probabilities = torch.softmax(outputs.logits, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

                # Get sentiment mapping based on number of classes
                if num_classes == 3:
                    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
                else:
                    # Generic mapping for unknown number of classes
                    sentiment_map = {i: f"Class_{i}" for i in range(num_classes)}

                sentiment = sentiment_map[predicted.item()]
                confidence_score = confidence.item()

            return sentiment, confidence_score

        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)

    except ImportError as e:
        st.error(f"Required library not installed: {str(e)}")
        st.info("Please install: pip install librosa transformers")
        return "Library not available", 0.0
    except Exception as e:
        st.error(f"Error in audio sentiment prediction: {str(e)}")
        return "Error occurred", 0.0


def predict_vision_sentiment(image, crop_tightness=0.05):
    """
    Load ResNet-50 and run inference for vision sentiment analysis

    Args:
        image: Input image (PIL Image or numpy array)
        crop_tightness: Padding around face (0.0 = no padding, 0.3 = 30% padding)
    """
    if image is None:
        return "No image provided", 0.0

    try:
        # Load model if not already loaded
        model, device, num_classes = load_vision_model()
        if model is None:
            return "Model not loaded", 0.0

        # Preprocess image to match FER2013 format
        st.info(
            "Detecting face and preprocessing image to match training data format..."
        )
        preprocessed_image = detect_and_preprocess_face(image, crop_tightness=0.0)

        if preprocessed_image is None:
            return "Image preprocessing failed", 0.0

        # Show preprocessed image
        st.image(
            preprocessed_image,
            caption="Preprocessed Image (48x48 Grayscale → 3-channel RGB)",
            width=200,
        )

        # Get transforms
        transform = get_vision_transforms()

        # Convert preprocessed image to tensor
        image_tensor = transform(preprocessed_image).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)

            # Debug: print output shape
            st.info(f"Model output shape: {outputs.shape}")

            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            # Get sentiment mapping based on number of classes
            sentiment_map = get_sentiment_mapping(num_classes)
            sentiment = sentiment_map[predicted.item()]
            confidence_score = confidence.item()

        return sentiment, confidence_score

    except Exception as e:
        st.error(f"Error in vision sentiment prediction: {str(e)}")
        st.error(
            f"Model output shape mismatch. Expected {num_classes} classes but got different."
        )
        return "Error occurred", 0.0


def predict_fused_sentiment(text=None, audio_bytes=None, image=None):
    """
    TODO: Implement ensemble/fusion logic combining all three models
    This is a placeholder function for fused sentiment analysis
    """
    # Placeholder logic - replace with actual fusion implementation
    results = []

    if text:
        text_sentiment, text_conf = predict_text_sentiment(text)
        results.append((text_sentiment, text_conf))

    if audio_bytes:
        audio_sentiment, audio_conf = predict_audio_sentiment(audio_bytes)
        results.append((audio_sentiment, audio_conf))

    if image:
        vision_sentiment, vision_conf = predict_vision_sentiment(image)
        results.append((vision_sentiment, vision_conf))

    if not results:
        return "No inputs provided", 0.0

    # Simple ensemble logic (replace with your fusion strategy)
    sentiment_counts = {}
    total_confidence = 0

    for sentiment, confidence in results:
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        total_confidence += confidence

    # Majority voting with confidence averaging
    final_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    avg_confidence = total_confidence / len(results)

    return final_sentiment, avg_confidence


# Sidebar navigation
st.sidebar.title("Sentiment Analysis")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox(
    "Choose a page:",
    [
        "Home",
        "Text Sentiment",
        "Audio Sentiment",
        "Vision Sentiment",
        "Fused Model",
    ],
)

# Home Page
if page == "Home":
    st.markdown(
        '<h1 class="main-header">Multimodal Sentiment Analysis</h1>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="model-card">
        <h2>Welcome to your Multi-Modal Sentiment Analysis Testing Platform!</h2>
        <p>This application provides a comprehensive testing environment for your three independent sentiment analysis models:</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class="model-card">
            <h3>Text Sentiment Model</h3>
            <p>READY TO USE - Analyze sentiment from text input using TextBlob</p>
                         <ul>
                 <li>Process any text input</li>
                 <li>Get sentiment classification (Positive/Negative/Neutral)</li>
                 <li>View confidence scores</li>
                 <li>Real-time NLP analysis</li>
             </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="model-card">
            <h3>Audio Sentiment Model</h3>
            <p>READY TO USE - Analyze sentiment from audio files using fine-tuned Wav2Vec2</p>
                         <ul>
                 <li>Upload audio files (.wav, .mp3, .m4a, .flac)</li>
                 <li>Record audio directly with microphone (max 5s)</li>
                 <li>Automatic preprocessing: 16kHz sampling, 5s max duration (CREMA-D + RAVDESS format)</li>
                 <li>Listen to uploaded/recorded audio</li>
                 <li>Get sentiment predictions</li>
                 <li>Real-time audio analysis</li>
             </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="model-card">
            <h3>Vision Sentiment Model</h3>
            <p>Analyze sentiment from images using fine-tuned ResNet-50</p>
                         <ul>
                 <li>Upload image files (.png, .jpg, .jpeg, .bmp, .tiff)</li>
                 <li>Automatic face detection & preprocessing</li>
                 <li>Fixed 0% padding for tightest face crop</li>
                 <li>Convert to 224x224 grayscale → 3-channel RGB (FER2013 format)</li>
                 <li>Transforms: Resize(224) → CenterCrop(224) → ImageNet Normalization</li>
                 <li>Preview original & preprocessed images</li>
                 <li>Get sentiment predictions</li>
             </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
    <div class="model-card">
        <h3>Fused Model</h3>
        <p>Combine predictions from all three models for enhanced accuracy</p>
        <ul>
            <li>Multi-modal input processing</li>
            <li>Ensemble prediction strategies</li>
            <li>Comprehensive sentiment analysis</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666;">
        <p><strong>Note:</strong> This application now has <strong>ALL THREE MODELS</strong> fully integrated and ready to use!</p>
        <p><strong>TextBlob</strong> (Text) + <strong>Wav2Vec2</strong> (Audio) + <strong>ResNet-50</strong> (Vision)</p>
        <p><strong>Models are now loaded from Google Drive automatically!</strong></p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# Text Sentiment Page
elif page == "Text Sentiment":
    st.title("Text Sentiment Analysis")
    st.markdown("Analyze the sentiment of your text using our TextBlob-based model.")

    # Text input
    text_input = st.text_area(
        "Enter your text here:",
        height=150,
        placeholder="Type or paste your text here to analyze its sentiment...",
    )

    # Analyze button
    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        if text_input and text_input.strip():
            with st.spinner("Analyzing text sentiment..."):
                sentiment, confidence = predict_text_sentiment(text_input)

                # Display results
                st.markdown("### Results")

                # Display results in columns
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sentiment", sentiment)
                with col2:
                    st.metric("Confidence", f"{confidence:.2f}")

                # Color-coded sentiment display
                sentiment_colors = {
                    "Positive": "🟢",
                    "Negative": "🔴",
                    "Neutral": "🟡",
                }

                st.markdown(
                    f"""
                <div class="result-box">
                    <h4>{sentiment_colors.get(sentiment, "❓")} Sentiment: {sentiment}</h4>
                    <p><strong>Confidence:</strong> {confidence:.2f}</p>
                    <p><strong>Input Text:</strong> "{text_input[:100]}{'...' if len(text_input) > 100 else ''}"</p>
                    <p><strong>Model:</strong> TextBlob (Natural Language Processing)</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.error("Please enter some text to analyze.")

# Audio Sentiment Page
elif page == "Audio Sentiment":
    st.title("Audio Sentiment Analysis")
    st.markdown(
        "Analyze the sentiment of your audio files using our fine-tuned Wav2Vec2 model."
    )

    # Preprocessing information
    st.info(
        "**Audio Preprocessing**: Audio will be automatically processed to match CREMA-D + RAVDESS training format: "
        "16kHz sampling rate, max 5 seconds, with automatic resampling and feature extraction."
    )

    # Model status
    model, device, num_classes, feature_extractor = load_audio_model()
    if model is None:
        st.error(
            "Audio model could not be loaded. Please check the Google Drive setup."
        )
        st.info(
            "Expected: Models should be configured in Google Drive and accessible via the model manager."
        )
    else:
        st.success(
            f"Audio model loaded successfully on {device} with {num_classes} classes!"
        )

    # Input method selection
    st.subheader("Choose Input Method")
    input_method = st.radio(
        "Select how you want to provide audio:",
        ["Upload Audio File", "Record Audio"],
        horizontal=True,
    )

    if input_method == "Upload Audio File":
        # File uploader
        uploaded_audio = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "m4a", "flac"],
            help="Supported formats: WAV, MP3, M4A, FLAC",
        )

        audio_source = "uploaded_file"
        audio_name = uploaded_audio.name if uploaded_audio else None

    else:  # Audio recording
        st.markdown(
            """
        <div class="model-card">
            <h3>Audio Recording</h3>
            <p>Record audio directly with your microphone (max 5 seconds).</p>
            <p><strong>Note:</strong> Make sure your microphone is accessible and you have permission to use it.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Audio recorder
        recorded_audio = st.audio_input(
            label="Click to start recording",
            help="Click the microphone button to start/stop recording. Maximum recording time is 5 seconds.",
        )

        if recorded_audio is not None:
            # Display recorded audio
            st.audio(recorded_audio, format="audio/wav")
            st.success("Audio recorded successfully!")

            # Convert recorded audio to bytes for processing
            uploaded_audio = recorded_audio
            audio_source = "recorded"
            audio_name = "Recorded Audio"
        else:
            uploaded_audio = None
            audio_source = None
            audio_name = None

    if uploaded_audio is not None:
        # Display audio player
        if audio_source == "recorded":
            st.audio(uploaded_audio, format="audio/wav")
            st.info(f"{audio_name} | Source: Microphone Recording")
        else:
            st.audio(
                uploaded_audio, format=f'audio/{uploaded_audio.name.split(".")[-1]}'
            )
            # File info for uploaded files
            file_size = len(uploaded_audio.getvalue()) / 1024  # KB
            st.info(f"File: {uploaded_audio.name} | Size: {file_size:.1f} KB")

        # Analyze button
        if st.button(
            "Analyze Audio Sentiment", type="primary", use_container_width=True
        ):
            if model is None:
                st.error("Model not loaded. Cannot analyze audio.")
            else:
                with st.spinner("Analyzing audio sentiment..."):
                    audio_bytes = uploaded_audio.getvalue()
                    sentiment, confidence = predict_audio_sentiment(audio_bytes)

                # Display results
                st.markdown("### Results")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sentiment", sentiment)
                with col2:
                    st.metric("Confidence", f"{confidence:.2f}")

                # Color-coded sentiment display
                sentiment_colors = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}

                st.markdown(
                    f"""
                <div class="result-box">
                    <h4>{sentiment_colors.get(sentiment, "❓")} Sentiment: {sentiment}</h4>
                    <p><strong>Confidence:</strong> {confidence:.2f}</p>
                    <p><strong>Audio Source:</strong> {audio_name}</p>
                    <p><strong>Model:</strong> Wav2Vec2 (Fine-tuned on RAVDESS + CREMA-D)</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
    else:
        if input_method == "Upload Audio File":
            st.info("Please upload an audio file to begin analysis.")
        else:
            st.info("Click the microphone button above to record audio for analysis.")

# Vision Sentiment Page
elif page == "Vision Sentiment":
    st.title("Vision Sentiment Analysis")
    st.markdown(
        "Analyze the sentiment of your images using our fine-tuned ResNet-50 model."
    )

    st.info(
        "**Note**: Images will be automatically preprocessed to match FER2013 format: face detection, grayscale conversion, and 224x224 resize (converted to 3-channel RGB)."
    )

    # Face cropping is set to 0% (no padding) for tightest crop
    st.info("**Face Cropping**: Set to 0% padding for tightest crop on facial features")

    # Model status
    model, device, num_classes = load_vision_model()
    if model is None:
        st.error(
            "Vision model could not be loaded. Please check the Google Drive setup."
        )
        st.info(
            "Expected: Models should be configured in Google Drive and accessible via the model manager."
        )
    else:
        st.success(
            f"Vision model loaded successfully on {device} with {num_classes} classes!"
        )

    # Input method selection
    st.subheader("Choose Input Method")
    input_method = st.radio(
        "Select how you want to provide an image:",
        ["Upload Image File", "Take Photo with Camera"],
        horizontal=True,
    )

    if input_method == "Upload Image File":
        # File uploader
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            help="Supported formats: PNG, JPG, JPEG, BMP, TIFF",
        )

        if uploaded_image is not None:
            # Display image
            image = Image.open(uploaded_image)
            st.image(
                image,
                caption=f"Uploaded Image: {uploaded_image.name}",
                use_container_width=True,
            )

            # File info
            file_size = len(uploaded_image.getvalue()) / 1024  # KB
            st.info(
                f"File: {uploaded_image.name} | Size: {file_size:.1f} KB | Dimensions: {image.size[0]}x{image.size[1]}"
            )

            # Analyze button
            if st.button(
                "Analyze Image Sentiment", type="primary", use_container_width=True
            ):
                if model is None:
                    st.error("Model not loaded. Cannot analyze image.")
                else:
                    with st.spinner("Analyzing image sentiment..."):
                        sentiment, confidence = predict_vision_sentiment(image)

                        # Display results
                        st.markdown("### Results")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Sentiment", sentiment)
                        with col2:
                            st.metric("Confidence", f"{confidence:.2f}")

                        # Color-coded sentiment display
                        sentiment_colors = {
                            "Positive": "🟢",
                            "Negative": "🔴",
                            "Neutral": "🟡",
                        }

                        st.markdown(
                            f"""
                        <div class="result-box">
                            <h4>{sentiment_colors.get(sentiment, "❓")} Sentiment: {sentiment}</h4>
                            <p><strong>Confidence:</strong> {confidence:.2f}</p>
                            <p><strong>Image File:</strong> {uploaded_image.name}</p>
                            <p><strong>Model:</strong> ResNet-50 (Fine-tuned on FER2013)</p>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

    else:  # Camera capture
        st.markdown(
            """
        <div class="model-card">
            <h3>Camera Capture</h3>
            <p>Take a photo directly with your camera to analyze its sentiment.</p>
            <p><strong>Note:</strong> Make sure your camera is accessible and you have permission to use it.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Camera input
        camera_photo = st.camera_input(
            "Take a photo",
            help="Click the camera button to take a photo, or use the upload button to select an existing photo",
        )

        if camera_photo is not None:
            # Display captured image
            image = Image.open(camera_photo)
            st.image(
                image,
                caption="Captured Photo",
                use_container_width=True,
            )

            # Image info
            st.info(
                f"Captured Photo | Dimensions: {image.size[0]}x{image.size[1]} | Format: {image.format}"
            )

            # Analyze button
            if st.button(
                "Analyze Photo Sentiment", type="primary", use_container_width=True
            ):
                if model is None:
                    st.error("Model not loaded. Cannot analyze image.")
                else:
                    with st.spinner("Analyzing photo sentiment..."):
                        sentiment, confidence = predict_vision_sentiment(image)

                        # Display results
                        st.markdown("### Results")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Sentiment", sentiment)
                        with col2:
                            st.metric("Confidence", f"{confidence:.2f}")

                        # Color-coded sentiment display
                        sentiment_colors = {
                            "Positive": "🟢",
                            "Negative": "🔴",
                            "Neutral": "🟡",
                        }

                        st.markdown(
                            f"""
                        <div class="result-box">
                            <h4>{sentiment_colors.get(sentiment, "❓")} Sentiment: {sentiment}</h4>
                            <p><strong>Confidence:</strong> {confidence:.2f}</p>
                            <p><strong>Image Source:</strong> Camera Capture</p>
                            <p><strong>Model:</strong> ResNet-50 (Fine-tuned on FER2013)</p>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

    # Show info if no image is provided
    if input_method == "Upload Image File" and "uploaded_image" not in locals():
        st.info("Please upload an image file to begin analysis.")
    elif input_method == "Take Photo with Camera" and "camera_photo" not in locals():
        st.info("Click the camera button above to take a photo for analysis.")

# Fused Model Page
elif page == "Fused Model":
    st.title("Fused Model Analysis")
    st.markdown(
        "Combine predictions from all three models for enhanced sentiment analysis."
    )

    st.markdown(
        """
    <div class="model-card">
        <h3>Multi-Modal Sentiment Analysis</h3>
        <p>This page allows you to input text, audio, and/or image data to get a comprehensive sentiment analysis 
        using all three models combined.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Input sections
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Text Input")
        text_input = st.text_area(
            "Enter text (optional):",
            height=100,
            placeholder="Type or paste your text here...",
        )

        st.subheader("Audio Input")

        # Audio preprocessing information for fused model
        st.info(
            "**Audio Preprocessing**: Audio will be automatically processed to match CREMA-D + RAVDESS training format: "
            "16kHz sampling rate, max 5 seconds, with automatic resampling and feature extraction."
        )

        # Audio input method for fused model
        audio_input_method = st.radio(
            "Audio input method:",
            ["Upload File", "Record Audio"],
            key="fused_audio_method",
            horizontal=True,
        )

        if audio_input_method == "Upload File":
            uploaded_audio = st.file_uploader(
                "Upload audio file (optional):",
                type=["wav", "mp3", "m4a", "flac"],
                key="fused_audio",
            )
            audio_source = "uploaded_file"
            audio_name = uploaded_audio.name if uploaded_audio else None
        else:
            # Audio recorder for fused model
            recorded_audio = st.audio_input(
                label="Record audio (optional):",
                key="fused_audio_recorder",
                help="Click to record audio for sentiment analysis",
            )

            if recorded_audio is not None:
                st.audio(recorded_audio, format="audio/wav")
                st.success("Audio recorded successfully!")
                uploaded_audio = recorded_audio
                audio_source = "recorded"
                audio_name = "Recorded Audio"
            else:
                uploaded_audio = None
                audio_source = None
                audio_name = None

    with col2:
        st.subheader("Image Input")

        # Face cropping is set to 0% (no padding) for tightest crop
        st.info(
            "**Face Cropping**: Set to 0% padding for tightest crop on facial features"
        )

        # Image input method for fused model
        image_input_method = st.radio(
            "Image input method:",
            ["Upload File", "Take Photo"],
            key="fused_image_method",
            horizontal=True,
        )

        if image_input_method == "Upload File":
            uploaded_image = st.file_uploader(
                "Upload image file (optional):",
                type=["png", "jpg", "jpeg", "bmp", "tiff"],
                key="fused_image",
            )

            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_container_width=True)
        else:
            # Camera capture for fused model
            camera_photo = st.camera_input(
                "Take a photo (optional):",
                key="fused_camera",
                help="Click to take a photo for sentiment analysis",
            )

            if camera_photo:
                image = Image.open(camera_photo)
                st.image(image, caption="Captured Photo", use_container_width=True)
                # Set uploaded_image to camera_photo for processing
                uploaded_image = camera_photo

        if uploaded_audio:
            st.audio(
                uploaded_audio, format=f'audio/{uploaded_audio.name.split(".")[-1]}'
            )

    # Analyze button
    if st.button("Run Fused Analysis", type="primary", use_container_width=True):
        if text_input or uploaded_audio or uploaded_image:
            with st.spinner("Running fused sentiment analysis..."):
                # Prepare inputs
                audio_bytes = uploaded_audio.getvalue() if uploaded_audio else None
                image = Image.open(uploaded_image) if uploaded_image else None

                # Get fused prediction
                sentiment, confidence = predict_fused_sentiment(
                    text=text_input if text_input else None,
                    audio_bytes=audio_bytes,
                    image=image,
                )

                # Display results
                st.markdown("### Fused Model Results")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Final Sentiment", sentiment)
                with col2:
                    st.metric("Overall Confidence", f"{confidence:.2f}")

                # Show individual model results
                st.markdown("### Individual Model Results")

                results_data = []

                if text_input:
                    text_sentiment, text_conf = predict_text_sentiment(text_input)
                    results_data.append(
                        {
                            "Model": "Text (TextBlob)",
                            "Input": f"Text: {text_input[:50]}...",
                            "Sentiment": text_sentiment,
                            "Confidence": f"{text_conf:.2f}",
                        }
                    )

                if uploaded_audio:
                    audio_sentiment, audio_conf = predict_audio_sentiment(audio_bytes)
                    results_data.append(
                        {
                            "Model": "Audio (Wav2Vec2)",
                            "Input": f"Audio: {audio_name}",
                            "Sentiment": audio_sentiment,
                            "Confidence": f"{audio_conf:.2f}",
                        }
                    )

                if uploaded_image:
                    # Face cropping is set to 0% (no padding) for tightest crop
                    vision_sentiment, vision_conf = predict_vision_sentiment(
                        image, crop_tightness=0.0
                    )
                    results_data.append(
                        {
                            "Model": "Vision (ResNet-50)",
                            "Input": f"Image: {uploaded_image.name}",
                            "Sentiment": vision_sentiment,
                            "Confidence": f"{vision_conf:.2f}",
                        }
                    )

                if results_data:
                    df = pd.DataFrame(results_data)
                    st.dataframe(df, use_container_width=True)

                # Final result display
                sentiment_colors = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}

                st.markdown(
                    f"""
                <div class="result-box">
                    <h4>{sentiment_colors.get(sentiment, "❓")} Final Fused Sentiment: {sentiment}</h4>
                    <p><strong>Overall Confidence:</strong> {confidence:.2f}</p>
                    <p><strong>Models Used:</strong> {len(results_data)}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning(
                "Please provide at least one input (text, audio, or image) for fused analysis."
            )

# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Built with ❤️ | by <a href="https://github.com/iamfaham">iamfaham</a></p>
</div>
""",
    unsafe_allow_html=True,
)
