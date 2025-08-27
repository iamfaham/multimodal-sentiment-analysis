"""
Refactored Sentiment Fused - Multimodal Sentiment Analysis Application

This is the main entry point for the application, now using a modular structure.
"""

import streamlit as st
import pandas as pd
from PIL import Image
import logging

# Import our modular components
from src.config.settings import (
    APP_NAME,
    APP_VERSION,
    APP_ICON,
    APP_LAYOUT,
    CUSTOM_CSS,
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
)
from src.models.text_model import predict_text_sentiment
from src.models.audio_model import predict_audio_sentiment, load_audio_model
from src.models.vision_model import predict_vision_sentiment, load_vision_model
from src.models.fused_model import predict_fused_sentiment
from src.utils.preprocessing import (
    extract_frames_from_video,
    extract_audio_from_video,
    transcribe_audio,
)
from src.utils.file_handling import get_file_info, format_file_size
from src.utils.sentiment_mapping import get_sentiment_colors, format_sentiment_result

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded",
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_home_page():
    """Render the home page with model information."""
    st.markdown(
        f'<h1 class="main-header">{APP_NAME}</h1>',
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
                    <li>Automatic preprocessing: 16kHz sampling, 5s max duration</li>
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

    st.markdown(
        """
        <div class="model-card">
            <h3>🎬 Max Fusion</h3>
            <p>Ultimate video-based sentiment analysis combining all three modalities</p>
            <ul>
                <li>🎥 Record or upload 5-second videos</li>
                <li>🔍 Extract frames for vision analysis</li>
                <li>🎵 Extract audio for vocal sentiment</li>
                <li>📝 Transcribe audio for text analysis</li>
                <li>🚀 Comprehensive multi-modal results</li>
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


def render_text_sentiment_page():
    """Render the text sentiment analysis page."""
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
                sentiment_colors = get_sentiment_colors()
                emoji = sentiment_colors.get(sentiment, "❓")

                st.markdown(
                    f"""
                    <div class="result-box">
                        <h4>{emoji} Sentiment: {sentiment}</h4>
                        <p><strong>Confidence:</strong> {confidence:.2f}</p>
                        <p><strong>Input Text:</strong> "{text_input[:100]}{'...' if len(text_input) > 100 else ''}"</p>
                        <p><strong>Model:</strong> TextBlob (Natural Language Processing)</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.error("Please enter some text to analyze.")


def render_audio_sentiment_page():
    """Render the audio sentiment analysis page."""
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
            type=SUPPORTED_AUDIO_FORMATS,
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
            file_info = get_file_info(uploaded_audio)
            st.info(
                f"File: {file_info['name']} | Size: {format_file_size(file_info['size_bytes'])}"
            )

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
                sentiment_colors = get_sentiment_colors()
                emoji = sentiment_colors.get(sentiment, "❓")

                st.markdown(
                    f"""
                    <div class="result-box">
                        <h4>{emoji} Sentiment: {sentiment}</h4>
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


def render_vision_sentiment_page():
    """Render the vision sentiment analysis page."""
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
            type=SUPPORTED_IMAGE_FORMATS,
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
            file_info = get_file_info(uploaded_image)
            st.info(
                f"File: {file_info['name']} | Size: {format_file_size(file_info['size_bytes'])} | Dimensions: {image.size[0]}x{image.size[1]}"
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
                        sentiment_colors = get_sentiment_colors()
                        emoji = sentiment_colors.get(sentiment, "❓")

                        st.markdown(
                            f"""
                            <div class="result-box">
                                <h4>{emoji} Sentiment: {sentiment}</h4>
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
                        sentiment_colors = get_sentiment_colors()
                        emoji = sentiment_colors.get(sentiment, "❓")

                        st.markdown(
                            f"""
                            <div class="result-box">
                                <h4>{emoji} Sentiment: {sentiment}</h4>
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


def render_fused_model_page():
    """Render the fused model analysis page."""
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
                type=SUPPORTED_AUDIO_FORMATS,
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
                type=SUPPORTED_IMAGE_FORMATS,
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
                sentiment_colors = get_sentiment_colors()
                emoji = sentiment_colors.get(sentiment, "❓")

                st.markdown(
                    f"""
                    <div class="result-box">
                        <h4>{emoji} Final Fused Sentiment: {sentiment}</h4>
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


def render_max_fusion_page():
    """Render the max fusion page for video-based analysis."""
    st.title("Max Fusion - Multi-Modal Sentiment Analysis")
    st.markdown(
        """
        <div class="model-card">
            <h3>Ultimate Multi-Modal Sentiment Analysis</h3>
            <p>Take photos with camera or upload videos to get comprehensive sentiment analysis from multiple modalities:</p>
            <ul>
                <li>📸 <strong>Vision Analysis:</strong> Camera photos or video frames for facial expression analysis</li>
                <li>🎵 <strong>Audio Analysis:</strong> Audio files or extracted audio from videos for vocal sentiment</li>
                <li>📝 <strong>Text Analysis:</strong> Transcribed audio for text sentiment analysis</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Video input method selection
    st.subheader("Video Input")
    video_input_method = st.radio(
        "Choose input method:",
        ["Upload Video File", "Record Video (Coming Soon)"],
        horizontal=True,
        index=0,  # Default to upload video
    )

    if video_input_method == "Record Video (Coming Soon)":
        # Coming Soon message for video recording
        st.info("🎥 Video recording feature is coming soon!")
        st.info("📁 Please use the Upload Video File option for now.")

        # Show a nice coming soon message
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                """
                <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
                    <h3>🚧 Coming Soon 🚧</h3>
                    <p>Video recording feature is under development</p>
                    <p>Use Upload Video File for now!</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Placeholder for future recording functionality
        st.markdown(
            """
            **Future Features:**
            - Real-time video recording with camera
            - Audio capture during recording
            - Automatic frame extraction
            - Live transcription
            - WebRTC integration for low-latency streaming
            """
        )

        # Skip all the recording logic for now
        uploaded_video = None
        video_source = None
        video_name = None
        video_file = None

    elif video_input_method == "Upload Video File":
        # File upload option
        st.markdown(
            """
            <div class="upload-section">
                <h4>📁 Upload Video File</h4>
                <p>Upload a video file for comprehensive multimodal analysis.</p>
                <p><strong>Supported Formats:</strong> MP4, AVI, MOV, MKV, WMV, FLV</p>
                <p><strong>Recommended:</strong> Videos with clear audio and visual content</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=SUPPORTED_VIDEO_FORMATS,
            help="Supported formats: MP4, AVI, MOV, MKV, WMV, FLV",
        )

        video_source = "uploaded_file"
        video_name = uploaded_video.name if uploaded_video else None
        video_file = uploaded_video

    if video_file is not None:
        # Display video or photo
        if video_source == "camera_photo":
            # For camera photos, we already displayed the image above
            st.info(f"Source: Camera Photo | Ready for vision analysis")

            # Add audio upload option for camera photo mode
            st.subheader("🎵 Audio Input for Analysis")
            st.info(
                "Since we're using a photo, please upload an audio file for audio sentiment analysis:"
            )

            uploaded_audio = st.file_uploader(
                "Upload audio file for audio analysis:",
                type=SUPPORTED_AUDIO_FORMATS,
                key="camera_audio",
                help="Upload an audio file to complement the photo analysis",
            )

            if uploaded_audio:
                st.audio(
                    uploaded_audio, format=f'audio/{uploaded_audio.name.split(".")[-1]}'
                )
                st.success("✅ Audio uploaded successfully!")
                audio_bytes = uploaded_audio.getvalue()
            else:
                audio_bytes = None
                st.warning("⚠️ Please upload an audio file for complete analysis")

        else:
            # For uploaded videos
            st.video(video_file)
            file_info = get_file_info(video_file)
            st.info(
                f"File: {file_info['name']} | Size: {format_file_size(file_info['size_bytes'])}"
            )
            audio_bytes = None  # Will be extracted from video

        # Video Processing Pipeline
        st.subheader("🎬 Video Processing Pipeline")

        # Initialize variables
        frames = []
        audio_bytes = None
        transcribed_text = ""

        # Process uploaded video
        if uploaded_video:
            st.info("📁 Processing uploaded video file...")

            # Extract frames
            st.markdown("**1. 🎥 Frame Extraction**")
            frames = extract_frames_from_video(uploaded_video, max_frames=5)

            if frames:
                st.success(f"✅ Extracted {len(frames)} representative frames")

                # Display extracted frames
                cols = st.columns(len(frames))
                for i, frame in enumerate(frames):
                    with cols[i]:
                        st.image(
                            frame, caption=f"Frame {i+1}", use_container_width=True
                        )
            else:
                st.warning("⚠️ Could not extract frames from video")
                frames = []

            # Extract audio
            st.markdown("**2. 🎵 Audio Extraction**")
            audio_bytes = extract_audio_from_video(uploaded_video)

            if audio_bytes:
                st.success("✅ Audio extracted successfully")
                st.audio(audio_bytes, format="audio/wav")
            else:
                st.warning("⚠️ Could not extract audio from video")
                audio_bytes = None

            # Transcribe audio
            st.markdown("**3. 📝 Audio Transcription**")
            if audio_bytes:
                transcribed_text = transcribe_audio(audio_bytes)
                if transcribed_text:
                    st.success("✅ Audio transcribed successfully")
                    st.markdown(f'**Transcribed Text:** "{transcribed_text}"')
                else:
                    st.warning("⚠️ Could not transcribe audio")
                    transcribed_text = ""
            else:
                transcribed_text = ""
                st.info("ℹ️ No audio available for transcription")

        # Analysis button
        if st.button(
            "🚀 Run Max Fusion Analysis", type="primary", use_container_width=True
        ):
            with st.spinner(
                "🔄 Processing video and running comprehensive analysis..."
            ):
                # Run individual analyses
                st.subheader("🔍 Individual Model Analysis")

                results_data = []

                # Vision analysis (use first frame for uploaded videos)
                if frames:
                    st.markdown("**Vision Analysis:**")

                    # For uploaded videos, use first frame
                    vision_sentiment, vision_conf = predict_vision_sentiment(
                        frames[0], crop_tightness=0.0
                    )
                    results_data.append(
                        {
                            "Model": "Vision (ResNet-50)",
                            "Input": f"Video Frame 1",
                            "Sentiment": vision_sentiment,
                            "Confidence": f"{vision_conf:.2f}",
                        }
                    )
                    st.success(
                        f"Vision: {vision_sentiment} (Confidence: {vision_conf:.2f})"
                    )

                # Audio analysis
                if audio_bytes:
                    st.markdown("**Audio Analysis:**")
                    audio_sentiment, audio_conf = predict_audio_sentiment(audio_bytes)
                    results_data.append(
                        {
                            "Model": "Audio (Wav2Vec2)",
                            "Input": f"Video Audio",
                            "Sentiment": audio_sentiment,
                            "Confidence": f"{audio_conf:.2f}",
                        }
                    )
                    st.success(
                        f"Audio: {audio_sentiment} (Confidence: {audio_conf:.2f})"
                    )

                # Text analysis
                if transcribed_text:
                    st.markdown("**Text Analysis:**")
                    text_sentiment, text_conf = predict_text_sentiment(transcribed_text)
                    results_data.append(
                        {
                            "Model": "Text (TextBlob)",
                            "Input": f"Transcribed: {transcribed_text[:50]}...",
                            "Sentiment": text_sentiment,
                            "Confidence": f"{text_conf:.2f}",
                        }
                    )
                    st.success(f"Text: {text_sentiment} (Confidence: {text_conf:.2f})")

                # Run fused analysis
                st.subheader("🎯 Max Fusion Results")

                if results_data:
                    # Display results table
                    df = pd.DataFrame(results_data)
                    st.dataframe(df, use_container_width=True)

                    # Calculate fused sentiment
                    image_for_fusion = frames[0] if frames else None
                    sentiment, confidence = predict_fused_sentiment(
                        text=transcribed_text if transcribed_text else None,
                        audio_bytes=audio_bytes,
                        image=image_for_fusion,
                    )

                    # Display final results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🎯 Final Sentiment", sentiment)
                    with col2:
                        st.metric("📊 Overall Confidence", f"{confidence:.2f}")

                    # Color-coded sentiment display
                    sentiment_colors = get_sentiment_colors()
                    emoji = sentiment_colors.get(sentiment, "❓")

                    st.markdown(
                        f"""
                        <div class="result-box">
                            <h4>{emoji} Max Fusion Sentiment: {sentiment}</h4>
                            <p><strong>Overall Confidence:</strong> {confidence:.2f}</p>
                            <p><strong>Modalities Analyzed:</strong> {len(results_data)}</p>
                            <p><strong>Video Source:</strong> {video_name}</p>
                            <p><strong>Analysis Type:</strong> Comprehensive Multi-Modal Sentiment Analysis</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(
                        "❌ No analysis could be performed. Please check your video input."
                    )

    else:
        if video_input_method == "Record Video (Coming Soon)":
            st.info(
                "🎥 Video recording feature is coming soon! Please use Upload Video File for now."
            )
        else:
            st.info("📁 Please upload a video file to begin Max Fusion analysis.")


def main():
    """Main application function."""
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
            "Max Fusion",
        ],
    )

    # Page routing
    if page == "Home":
        render_home_page()
    elif page == "Text Sentiment":
        render_text_sentiment_page()
    elif page == "Audio Sentiment":
        render_audio_sentiment_page()
    elif page == "Vision Sentiment":
        render_vision_sentiment_page()
    elif page == "Fused Model":
        render_fused_model_page()
    elif page == "Max Fusion":
        render_max_fusion_page()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>Built with ❤️ | by <a href="https://github.com/iamfaham">iamfaham</a></p>
            <p>Version: {version}</p>
        </div>
        """.format(
            version=APP_VERSION
        ),
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
