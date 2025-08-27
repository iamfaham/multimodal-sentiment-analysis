"""
Centralized configuration settings for the Sentiment Fused application.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Application Configuration
APP_NAME = "Multimodal Sentiment Analysis"
APP_VERSION = "0.1.0"
APP_ICON = "🧠"
APP_LAYOUT = "wide"

# Model Configuration
VISION_MODEL_CONFIG = {
    "model_name": "resnet50",
    "input_size": 224,
    "num_classes": 7,  # FER2013 default
    "crop_tightness": 0.0,  # No padding for tightest crop
}

AUDIO_MODEL_CONFIG = {
    "model_name": "facebook/wav2vec2-base",
    "target_sampling_rate": 16000,
    "max_duration": 5.0,
    "num_classes": 3,  # Default sentiment classes
}

TEXT_MODEL_CONFIG = {
    "model_name": "textblob",
    "confidence_threshold": 0.1,
}

# File Processing Configuration
SUPPORTED_IMAGE_FORMATS = ["png", "jpg", "jpeg", "bmp", "tiff"]
SUPPORTED_AUDIO_FORMATS = ["wav", "mp3", "m4a", "flac"]
SUPPORTED_VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv", "wmv", "flv"]

# Video Processing Configuration
MAX_VIDEO_FRAMES = 5
VIDEO_FRAME_INTERVALS = [0, 0.25, 0.5, 0.75, 1.0]  # Frame extraction points

# Image Preprocessing Configuration
IMAGE_TRANSFORMS = {
    "resize": 224,
    "center_crop": 224,
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
}

# Sentiment Mapping Configuration
SENTIMENT_MAPPINGS = {
    3: {0: "Negative", 1: "Neutral", 2: "Positive"},
    4: {0: "Angry", 1: "Sad", 2: "Happy", 3: "Neutral"},
    7: {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral",
    },
}

# UI Configuration
UI_COLORS = {
    "primary": "#1f77b4",
    "success": "#28a745",
    "warning": "#ffc107",
    "danger": "#dc3545",
    "info": "#17a2b8",
    "light": "#f8f9fa",
    "dark": "#343a40",
}

# CSS Styles
CUSTOM_CSS = """
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
"""

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"
UI_DIR = SRC_DIR / "ui"

# Environment Variables
ENV_VARS = {
    "VISION_MODEL_DRIVE_ID": os.getenv("VISION_MODEL_DRIVE_ID", ""),
    "AUDIO_MODEL_DRIVE_ID": os.getenv("AUDIO_MODEL_DRIVE_ID", ""),
    "VISION_MODEL_FILENAME": os.getenv("VISION_MODEL_FILENAME", "resnet50_model.pth"),
    "AUDIO_MODEL_FILENAME": os.getenv("AUDIO_MODEL_FILENAME", "wav2vec2_model.pth"),
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": ["console", "file"],
}

# Cache Configuration
CACHE_CONFIG = {
    "ttl": 3600,  # 1 hour
    "max_entries": 100,
}


def get_sentiment_mapping(num_classes: int) -> Dict[int, str]:
    """Get sentiment mapping based on number of classes."""
    return SENTIMENT_MAPPINGS.get(
        num_classes, {i: f"Class_{i}" for i in range(num_classes)}
    )


def validate_environment() -> Dict[str, bool]:
    """Validate that required environment variables are set."""
    validation = {}
    for var_name, var_value in ENV_VARS.items():
        validation[var_name] = bool(var_value)
    return validation
