"""
Audio sentiment analysis model using fine-tuned Wav2Vec2.
"""

import logging
import streamlit as st
from typing import Tuple
import torch
from PIL import Image
import os
from ..config.settings import AUDIO_MODEL_CONFIG
from ..utils.preprocessing import preprocess_audio_for_model
from ..utils.sentiment_mapping import get_sentiment_mapping
from src.utils.simple_model_manager import SimpleModelManager

logger = logging.getLogger(__name__)


@st.cache_resource
def load_audio_model():
    """Load the pre-trained Wav2Vec2 audio sentiment model from Google Drive."""
    try:
        manager = SimpleModelManager()
        if manager is None:
            logger.error("Model manager not available")
            st.error("Model manager not available")
            return None, None, None, None

        # Load the model using the Google Drive manager
        model, device = manager.load_audio_model()

        if model is None:
            logger.error("Failed to load audio model from Google Drive")
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
                num_classes = AUDIO_MODEL_CONFIG["num_classes"]  # Default assumption

        # Load feature extractor
        from transformers import AutoFeatureExtractor

        feature_extractor = AutoFeatureExtractor.from_pretrained(
            AUDIO_MODEL_CONFIG["model_name"]
        )

        logger.info(f"Audio model loaded successfully with {num_classes} classes!")
        st.success(f"Audio model loaded successfully with {num_classes} classes!")
        return model, device, num_classes, feature_extractor
    except Exception as e:
        logger.error(f"Error loading audio model: {str(e)}")
        st.error(f"Error loading audio model: {str(e)}")
        return None, None, None, None


def predict_audio_sentiment(audio_bytes: bytes) -> Tuple[str, float]:
    """
    Analyze audio sentiment using fine-tuned Wav2Vec2 model.

    Preprocessing matches CREMA-D + RAVDESS training specifications:
    - Target sampling rate: 16kHz
    - Max duration: 5.0 seconds
    - Feature extraction: AutoFeatureExtractor with max_length, truncation, padding

    Args:
        audio_bytes: Raw audio bytes

    Returns:
        Tuple of (sentiment, confidence)
    """
    if audio_bytes is None:
        return "No audio provided", 0.0

    try:
        # Load model if not already loaded
        model, device, num_classes, feature_extractor = load_audio_model()
        if model is None:
            return "Model not loaded", 0.0

        # Use our centralized preprocessing function
        input_values = preprocess_audio_for_model(audio_bytes)
        if input_values is None:
            return "Preprocessing failed", 0.0

        # Debug: Log the tensor shape
        logger.info(f"Preprocessed audio tensor shape: {input_values.shape}")

        # Ensure correct tensor shape: [batch_size, sequence_length]
        if input_values.dim() == 1:
            input_values = input_values.unsqueeze(0)  # Add batch dimension if missing
        elif input_values.dim() == 3:
            # If we get [batch, sequence, channels], squeeze the channels
            input_values = input_values.squeeze(-1)

        logger.info(f"Final audio tensor shape: {input_values.shape}")

        # Move to device
        input_values = input_values.to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(input_values)
            probabilities = torch.softmax(outputs.logits, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            # Get sentiment mapping based on number of classes
            sentiment_map = get_sentiment_mapping(num_classes)
            sentiment = sentiment_map[predicted.item()]
            confidence_score = confidence.item()

        logger.info(
            f"Audio sentiment analysis completed: {sentiment} (confidence: {confidence_score:.2f})"
        )
        return sentiment, confidence_score

    except ImportError as e:
        logger.error(f"Required library not installed: {str(e)}")
        st.error(f"Required library not installed: {str(e)}")
        st.info("Please install: pip install librosa transformers")
        return "Library not available", 0.0
    except Exception as e:
        logger.error(f"Error in audio sentiment prediction: {str(e)}")
        st.error(f"Error in audio sentiment prediction: {str(e)}")
        return "Error occurred", 0.0


def get_audio_model_info() -> dict:
    """Get information about the audio sentiment model."""
    return {
        "model_name": AUDIO_MODEL_CONFIG["model_name"],
        "description": "Fine-tuned Wav2Vec2 for audio sentiment analysis",
        "capabilities": [
            "Audio sentiment classification",
            "Automatic audio preprocessing",
            "CREMA-D + RAVDESS dataset compatibility",
            "Real-time audio analysis",
        ],
        "input_format": "Audio files (WAV, MP3, M4A, FLAC)",
        "output_format": "Sentiment label + confidence score",
        "preprocessing": {
            "sampling_rate": f"{AUDIO_MODEL_CONFIG['target_sampling_rate']} Hz",
            "max_duration": f"{AUDIO_MODEL_CONFIG['max_duration']} seconds",
            "feature_extraction": "AutoFeatureExtractor",
            "normalization": "Model-specific",
        },
    }
