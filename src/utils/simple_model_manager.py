import os
import gdown
from pathlib import Path
import logging
from typing import Tuple, Any
import torch
import torch.nn as nn
from torchvision import models
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleModelManager:
    """Simple model manager that downloads models from Google Drive using gdown"""

    def __init__(self, model_dir: str = "model_weights", cache_models: bool = True):
        """
        Initialize simple model manager

        Args:
            model_dir: Local directory to store models
            cache_models: Whether to cache models locally
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.cache_models = cache_models

        # Load model links from environment variables
        self.model_links = {
            "vision": {
                "url": os.getenv("VISION_MODEL_DRIVE_ID", ""),
                "filename": os.getenv("VISION_MODEL_FILENAME", "resnet50_model.pth"),
                "description": "Vision sentiment analysis model",
            },
            "audio": {
                "url": os.getenv("AUDIO_MODEL_DRIVE_ID", ""),
                "filename": os.getenv("AUDIO_MODEL_FILENAME", "wav2vec2_model.pth"),
                "description": "Audio sentiment analysis model",
            },
        }

        # Validate that environment variables are set
        self._validate_environment()

    def _validate_environment(self):
        """Validate that required environment variables are set"""
        missing_vars = []

        if not self.model_links["vision"]["url"]:
            missing_vars.append("VISION_MODEL_DRIVE_ID")

        if not self.model_links["audio"]["url"]:
            missing_vars.append("AUDIO_MODEL_DRIVE_ID")

        if missing_vars:
            logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
            logger.warning("Please set these in your .env file or environment")
            logger.warning("Models will not be available until these are configured")

    def download_from_google_drive(self, share_url: str, filename: str) -> str:
        """
        Download file from Google Drive share link using gdown

        Args:
            share_url: Google Drive share link
            filename: Name to save the file as

        Returns:
            Path to downloaded file
        """
        try:
            local_path = self.model_dir / filename

            if local_path.exists() and self.cache_models:
                logger.info(f"Model already cached: {local_path}")
                return str(local_path)

            logger.info(f"Downloading {filename} from Google Drive using gdown...")

            # Use gdown to download the file
            # gdown automatically handles virus scan warnings and other Google Drive issues
            output_path = str(local_path)

            # Download with progress bar
            gdown.download(
                id=share_url,
                output=output_path,
                quiet=False,  # Show progress bar
                fuzzy=True,  # Handle various Google Drive URL formats
            )

            # Verify the file was downloaded
            if not Path(output_path).exists():
                raise FileNotFoundError(f"Download failed: {output_path} not found")

            file_size = Path(output_path).stat().st_size
            if file_size == 0:
                raise ValueError(f"Downloaded file is empty: {output_path}")

            logger.info(f"Successfully downloaded {filename} ({file_size} bytes)")
            return output_path

        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            raise

    def load_vision_model(self) -> Tuple[Any, torch.device, int]:
        """Load vision sentiment model"""
        try:
            model_info = self.model_links["vision"]

            # Check if URL is configured
            if not model_info["url"]:
                raise ValueError("VISION_MODEL_DRIVE_ID environment variable not set")

            model_path = self.download_from_google_drive(
                model_info["url"], model_info["filename"]
            )

            # Validate the downloaded file
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")

            file_size = Path(model_path).stat().st_size
            if file_size == 0:
                raise ValueError(f"Model file is empty: {model_path}")

            # Check file header to see what type of file it is
            with open(model_path, "rb") as f:
                header = f.read(100)  # Read first 100 bytes

            logger.info(f"File size: {file_size} bytes")
            logger.info(f"File header (first 100 bytes): {header[:50]}...")

            # Try to detect file type
            if header.startswith(b"<"):
                raise ValueError(
                    f"File appears to be HTML/XML, not a PyTorch model: {model_path}"
                )
            elif header.startswith(b"\x89PNG"):
                raise ValueError(f"File appears to be a PNG image: {model_path}")
            elif header.startswith(b"\xff\xd8\xff"):
                raise ValueError(f"File appears to be a JPEG image: {model_path}")

            # For any other file type (including ZIP), try to load it directly as a PyTorch model
            logger.info(
                f"File appears to be a PyTorch model file, attempting to load directly..."
            )

            # Load the model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                # Try loading the file directly as a PyTorch model
                checkpoint = torch.load(
                    model_path, map_location=device, weights_only=False
                )
                logger.info("Successfully loaded model file directly")
            except Exception as load_error:
                logger.error(f"Failed to load model directly: {load_error}")
                try:
                    # Try with weights only as fallback
                    checkpoint = torch.load(
                        model_path, map_location=device, weights_only=True
                    )
                    logger.info("Loaded with weights_only=True (weights only)")
                except Exception as fallback_error:
                    logger.error(
                        f"Failed to load with weights_only=True: {fallback_error}"
                    )
                    raise ValueError(
                        f"Cannot load model file {model_path}. File may be corrupted or in wrong format."
                    )

            # Initialize ResNet-50 model
            model = models.resnet50(weights=None)
            num_ftrs = model.fc.in_features

            # Determine number of classes from checkpoint
            if "fc.weight" in checkpoint:
                num_classes = checkpoint["fc.weight"].shape[0]
            else:
                num_classes = 3  # Default fallback

            model.fc = nn.Linear(num_ftrs, num_classes)
            model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()

            logger.info(f"Vision model loaded successfully with {num_classes} classes!")
            return model, device, num_classes

        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")
            raise

    def load_audio_model(self) -> Tuple[Any, torch.device]:
        """Load audio sentiment model"""
        try:
            model_info = self.model_links["audio"]

            # Check if URL is configured
            if not model_info["url"]:
                raise ValueError("AUDIO_MODEL_DRIVE_ID environment variable not set")

            model_path = self.download_from_google_drive(
                model_info["url"], model_info["filename"]
            )

            # Validate the downloaded file
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")

            file_size = Path(model_path).stat().st_size
            if file_size == 0:
                raise ValueError(f"Model file is empty: {model_path}")

            # Check file header to see what type of file it is
            with open(model_path, "rb") as f:
                header = f.read(100)  # Read first 100 bytes

            logger.info(f"File size: {file_size} bytes")
            logger.info(f"File header (first 100 bytes): {header[:50]}...")

            # Try to detect file type
            if header.startswith(b"<"):
                raise ValueError(
                    f"File appears to be HTML/XML, not a PyTorch model: {model_path}"
                )
            elif header.startswith(b"\x89PNG"):
                raise ValueError(f"File appears to be a PNG image: {model_path}")
            elif header.startswith(b"\xff\xd8\xff"):
                raise ValueError(f"File appears to be a JPEG image: {model_path}")

            # For any other file type (including ZIP), try to load it directly as a PyTorch model
            logger.info(
                f"File appears to be a PyTorch model file, attempting to load directly..."
            )

            # Load the model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                # Try loading the file directly as a PyTorch model
                checkpoint = torch.load(
                    model_path, map_location=device, weights_only=False
                )
                logger.info("Successfully loaded model file directly")
            except Exception as load_error:
                logger.error(f"Failed to load model directly: {load_error}")
                try:
                    # Try with weights only as fallback
                    checkpoint = torch.load(
                        model_path, map_location=device, weights_only=True
                    )
                    logger.info("Loaded with weights_only=True (weights only)")
                except Exception as fallback_error:
                    logger.error(
                        f"Failed to load with weights_only=True: {fallback_error}"
                    )
                    raise ValueError(
                        f"Cannot load model file {model_path}. File may be corrupted or in wrong format."
                    )

            # Check if we have a state dict or a full model
            if isinstance(checkpoint, dict) and "classifier.weight" in checkpoint:
                # This is a state dictionary - we need to initialize the model first
                from transformers import AutoModelForAudioClassification

                # Determine number of classes from checkpoint
                if "classifier.weight" in checkpoint:
                    num_classes = checkpoint["classifier.weight"].shape[0]
                else:
                    num_classes = 3  # Default fallback

                # Initialize Wav2Vec2 model with the correct number of classes
                model = AutoModelForAudioClassification.from_pretrained(
                    "facebook/wav2vec2-base", num_labels=num_classes
                )

                # Load the state dictionary
                model.load_state_dict(checkpoint)
                model.to(device)
                model.eval()

                logger.info(
                    f"Audio model loaded successfully with {num_classes} classes!"
                )
                return model, device
            else:
                # This is a full model object
                model = checkpoint
                model.to(device)
                model.eval()

                logger.info("Audio model loaded successfully!")
                return model, device

        except Exception as e:
            logger.error(f"Failed to load audio model: {e}")
            raise

    def update_model_links(self, vision_url: str = None, audio_url: str = None):
        """Update Google Drive URLs for models (optional override)"""
        if vision_url:
            self.model_links["vision"]["url"] = vision_url
        if audio_url:
            self.model_links["audio"]["url"] = audio_url

        # Update environment variables if provided
        if vision_url:
            os.environ["VISION_MODEL_DRIVE_ID"] = vision_url
        if audio_url:
            os.environ["AUDIO_MODEL_DRIVE_ID"] = audio_url

        logger.info("Model links updated!")

    def list_cached_models(self) -> list:
        """List all cached models"""
        cached_models = []
        for file_path in self.model_dir.glob("*.pth"):
            cached_models.append(file_path.name)
        return cached_models

    def clear_cache(self):
        """Clear all cached models"""
        for file_path in self.model_dir.glob("*.pth"):
            file_path.unlink()
        logger.info("Cache cleared!")

    def get_model_status(self) -> dict:
        """Get status of all models"""
        status = {}
        for model_type, info in self.model_links.items():
            status[model_type] = {
                "configured": bool(info["url"]),
                "filename": info["filename"],
                "cached": (self.model_dir / info["filename"]).exists(),
                "url": info["url"] if info["url"] else "Not configured",
            }
        return status


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = SimpleModelManager()

    # Check model status
    status = manager.get_model_status()
    print("Model Status:")
    for model_type, info in status.items():
        print(f"  {model_type}: {'✅' if info['configured'] else '❌'} {info['url']}")
        if info["cached"]:
            print(f"    📁 Cached: {info['filename']}")

    # Load models if configured
    try:
        if status["vision"]["configured"]:
            vision_model, device, num_classes = manager.load_vision_model()
            print(f"✅ Vision model loaded: {num_classes} classes")
        else:
            print("❌ Vision model not configured")

        if status["audio"]["configured"]:
            audio_model, device = manager.load_audio_model()
            print("✅ Audio model loaded")
        else:
            print("❌ Audio model not configured")

        if status["vision"]["configured"] and status["audio"]["configured"]:
            print("\n🎉 All models loaded successfully!")
        else:
            print("\n⚠️  Some models are not configured")
            print("Please set the following environment variables:")
            print("  VISION_MODEL_DRIVE_ID")
            print("  AUDIO_MODEL_DRIVE_ID")

    except Exception as e:
        print(f"Error loading models: {e}")
        print("\nFor folder structures:")
        print("   1. Navigate to each subfolder (Audio/Vision)")
        print("   2. Right-click on each .pth file")
        print("   3. Share -> Copy link")
        print("   4. Use those direct file links instead of folder links")
        print("\nNote: Downloaded files are used directly as PyTorch models.")
        print("\nOr set environment variables in your .env file:")
        print("  VISION_MODEL_DRIVE_ID=your_vision_model_file_id")
        print("  AUDIO_MODEL_DRIVE_ID=your_audio_model_file_id")
