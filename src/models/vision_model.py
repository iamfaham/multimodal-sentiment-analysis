"""
Vision sentiment analysis model using fine-tuned ResNet-50.
"""

import logging
import streamlit as st
from typing import Tuple
import torch
import torch.nn.functional as F
from PIL import Image

from ..config.settings import VISION_MODEL_CONFIG
from ..utils.preprocessing import detect_and_preprocess_face, get_vision_transforms
from ..utils.sentiment_mapping import get_sentiment_mapping
from src.utils.simple_model_manager import SimpleModelManager

logger = logging.getLogger(__name__)


@st.cache_resource
def get_model_manager():
    """Get the Google Drive model manager instance."""
    try:
        manager = SimpleModelManager()
        return manager
    except Exception as e:
        logger.error(f"Failed to initialize model manager: {e}")
        st.error(f"Failed to initialize model manager: {e}")
        return None


@st.cache_resource
def load_vision_model():
    """Load the pre-trained ResNet-50 vision sentiment model from Google Drive."""
    try:
        manager = get_model_manager()
        if manager is None:
            logger.error("Model manager not available")
            st.error("Model manager not available")
            return None, None, None

        # Load the model using the Google Drive manager
        model, device, num_classes = manager.load_vision_model()

        if model is None:
            logger.error("Failed to load vision model from Google Drive")
            st.error("Failed to load vision model from Google Drive")
            return None, None, None

        logger.info(f"Vision model loaded successfully with {num_classes} classes!")
        st.success(f"Vision model loaded successfully with {num_classes} classes!")
        return model, device, num_classes
    except Exception as e:
        logger.error(f"Error loading vision model: {str(e)}")
        st.error(f"Error loading vision model: {str(e)}")
        return None, None, None


def predict_vision_sentiment(
    image: Image.Image, crop_tightness: float = None
) -> Tuple[str, float]:
    """
    Load ResNet-50 and run inference for vision sentiment analysis.

    Args:
        image: Input image (PIL Image or numpy array)
        crop_tightness: Padding around face (0.0 = no padding, 0.3 = 30% padding)

    Returns:
        Tuple of (sentiment, confidence)
    """
    if image is None:
        return "No image provided", 0.0

    try:
        # Use default crop tightness if not specified
        if crop_tightness is None:
            crop_tightness = VISION_MODEL_CONFIG["crop_tightness"]

        # Load model if not already loaded
        model, device, num_classes = load_vision_model()
        if model is None:
            return "Model not loaded", 0.0

        # Preprocess image to match FER2013 format
        st.info(
            "Detecting face and preprocessing image to match training data format..."
        )
        preprocessed_image = detect_and_preprocess_face(
            image, crop_tightness=crop_tightness
        )

        if preprocessed_image is None:
            return "Image preprocessing failed", 0.0

        # Show preprocessed image
        st.image(
            preprocessed_image,
            caption="Preprocessed Image (224x224 Grayscale → 3-channel RGB)",
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

        logger.info(
            f"Vision sentiment analysis completed: {sentiment} (confidence: {confidence_score:.2f})"
        )
        return sentiment, confidence_score

    except Exception as e:
        logger.error(f"Error in vision sentiment prediction: {str(e)}")
        st.error(f"Error in vision sentiment prediction: {str(e)}")
        st.error(
            f"Model output shape mismatch. Expected {num_classes} classes but got different."
        )
        return "Error occurred", 0.0


def get_vision_model_info() -> dict:
    """Get information about the vision sentiment model."""
    return {
        "model_name": VISION_MODEL_CONFIG["model_name"],
        "description": "Fine-tuned ResNet-50 for facial expression sentiment analysis",
        "capabilities": [
            "Facial expression recognition",
            "Automatic face detection and cropping",
            "FER2013 dataset format compatibility",
            "Real-time image analysis",
        ],
        "input_format": "Images (PNG, JPG, JPEG, BMP, TIFF)",
        "output_format": "Sentiment label + confidence score",
        "preprocessing": {
            "face_detection": "OpenCV Haar Cascade",
            "image_size": f"{VISION_MODEL_CONFIG['input_size']}x{VISION_MODEL_CONFIG['input_size']}",
            "color_format": "Grayscale → 3-channel RGB",
            "normalization": "ImageNet standard",
        },
    }
