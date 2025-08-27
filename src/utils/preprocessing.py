"""
Preprocessing utilities for different input modalities.
"""

import os
import tempfile
import logging
from typing import List, Optional, Tuple, Union

try:
    from PIL import Image
    import numpy as np

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    np = None

from ..config.settings import (
    IMAGE_TRANSFORMS,
    AUDIO_MODEL_CONFIG,
)

# Add Any to typing imports
from typing import List, Optional, Union, Any

# Add torch import for audio preprocessing
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)


def detect_and_preprocess_face(
    image: Union[Image.Image, np.ndarray, Any], crop_tightness: float = 0.05
) -> Optional[Image.Image]:
    """
    Detect face in image, crop to face region, convert to grayscale, and resize.

    Args:
        image: Input image (PIL Image or numpy array)
        crop_tightness: Padding around face (0.0 = no padding, 0.3 = 30% padding)

    Returns:
        Preprocessed PIL Image or None if preprocessing fails
    """
    if not PIL_AVAILABLE:
        logger.error("PIL (Pillow) not available. Cannot process images.")
        return None

    try:
        import cv2

        # Convert PIL image to OpenCV format
        if isinstance(image, Image.Image):
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

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            logger.warning("No face detected in the image. Using center crop instead.")
            return _fallback_preprocessing(image)

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

        # Resize to target size
        target_size = IMAGE_TRANSFORMS["resize"]
        face_resized = cv2.resize(
            face_gray, (target_size, target_size), interpolation=cv2.INTER_AREA
        )

        # Convert grayscale to 3-channel RGB (replicate grayscale values)
        face_rgb_3channel = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)

        # Convert back to PIL Image
        face_pil = Image.fromarray(face_rgb_3channel)
        return face_pil

    except ImportError:
        logger.error(
            "OpenCV not installed. Please install it with: pip install opencv-python"
        )
        return _fallback_preprocessing(image)
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        return _fallback_preprocessing(image)


def _fallback_preprocessing(
    image: Union[Image.Image, np.ndarray, Any],
) -> Optional[Image.Image]:
    """Fallback preprocessing when face detection fails."""
    try:
        if isinstance(image, Image.Image):
            rgb_pil = image.convert("RGB")
            target_size = IMAGE_TRANSFORMS["resize"]
            resized = rgb_pil.resize(
                (target_size, target_size), Image.Resampling.LANCZOS
            )
            # Convert to grayscale and then to 3-channel RGB
            gray_pil = resized.convert("L")
            gray_rgb_pil = gray_pil.convert("RGB")
            return gray_rgb_pil
        return None
    except Exception as e:
        logger.error(f"Fallback preprocessing failed: {str(e)}")
        return None


def get_vision_transforms():
    """Get the image transforms used during training."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(IMAGE_TRANSFORMS["resize"]),
            transforms.CenterCrop(IMAGE_TRANSFORMS["center_crop"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGE_TRANSFORMS["normalize_mean"],
                std=IMAGE_TRANSFORMS["normalize_std"],
            ),
        ]
    )


def preprocess_audio_for_model(audio_bytes: bytes) -> Optional[torch.Tensor]:
    """
    Preprocess audio bytes for wav2vec2 model input using AutoFeatureExtractor.

    Args:
        audio_bytes: Raw audio bytes

    Returns:
        Preprocessed audio tensor ready for wav2vec2 model
    """
    if not TORCH_AVAILABLE:
        logger.error("PyTorch not available. Cannot process audio.")
        return None

    try:
        from transformers import AutoFeatureExtractor
        import librosa

        # Save audio bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name

        try:
            # Load and resample audio to target sampling rate
            audio, sr = librosa.load(
                tmp_file_path, sr=AUDIO_MODEL_CONFIG["target_sampling_rate"]
            )

            # Use AutoFeatureExtractor (same as training)
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                AUDIO_MODEL_CONFIG["model_name"]
            )

            # Calculate max length in samples (5 seconds * 16kHz = 80000 samples)
            max_length = int(
                AUDIO_MODEL_CONFIG["max_duration"]
                * AUDIO_MODEL_CONFIG["target_sampling_rate"]
            )

            logger.info(f"Audio length: {len(audio)} samples, max_length: {max_length}")

            inputs = feature_extractor(
                audio,
                sampling_rate=AUDIO_MODEL_CONFIG["target_sampling_rate"],
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            # Return tensor with correct shape for wav2vec2
            # The model expects: [batch_size, sequence_length]
            tensor = inputs.input_values

            # Log the tensor shape for debugging
            logger.info(f"Audio preprocessing output shape: {tensor.shape}")

            return tensor

        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except (OSError, PermissionError):
                pass

    except ImportError as e:
        logger.error(f"Required library not installed: {str(e)}")
        raise ImportError("Please install: pip install transformers librosa torch")


def extract_frames_from_video(video_file, max_frames: int = 5) -> List[Any]:
    """
    Extract frames from video file for vision sentiment analysis.

    Args:
        video_file: Video file object
        max_frames: Maximum number of frames to extract

    Returns:
        List of PIL Image objects
    """
    try:
        import cv2

        # Save video bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            if hasattr(video_file, "getvalue"):
                tmp_file.write(video_file.getvalue())
            else:
                tmp_file.write(video_file)
            tmp_file_path = tmp_file.name

        try:
            # Open video with OpenCV
            cap = cv2.VideoCapture(tmp_file_path)

            if not cap.isOpened():
                logger.error("Could not open video file")
                return []

            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            logger.info(
                f"Video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration"
            )

            # Extract frames at strategic intervals
            if total_frames > 0:
                # Select frames: start, 25%, 50%, 75%, end
                frame_indices = [
                    0,
                    int(total_frames * 0.25),
                    int(total_frames * 0.5),
                    int(total_frames * 0.75),
                    total_frames - 1,
                ]
                frame_indices = list(set(frame_indices))  # Remove duplicates
                frame_indices.sort()

                for frame_idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Convert to PIL Image
                        pil_image = Image.fromarray(frame_rgb)
                        frames.append(pil_image)

            cap.release()
            return frames

        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except (OSError, PermissionError):
                pass

    except ImportError:
        logger.error(
            "OpenCV not installed. Please install it with: pip install opencv-python"
        )
        return []
    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}")
        return []


def extract_audio_from_video(video_file) -> Optional[bytes]:
    """
    Extract audio from video file for audio sentiment analysis.

    Args:
        video_file: Video file object

    Returns:
        Audio bytes in WAV format or None if extraction fails
    """
    try:
        import tempfile

        try:
            from moviepy import VideoFileClip
        except ImportError as e:
            logger.error(f"MoviePy import failed: {e}")
            return None

        # Save video bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            if hasattr(video_file, "getvalue"):
                tmp_file.write(video_file.getvalue())
            else:
                tmp_file.write(video_file)
            tmp_file_path = tmp_file.name

        try:
            # Extract audio using moviepy
            video = VideoFileClip(tmp_file_path)
            audio = video.audio

            if audio is None:
                logger.warning("No audio track found in video")
                return None

            # Save audio to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
                audio_path = audio_file.name

            # Export audio as WAV
            audio.write_audiofile(audio_path, logger=None)

            # Read the audio file and return bytes
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            # Clean up temporary audio file
            try:
                os.unlink(audio_path)
            except (OSError, PermissionError):
                pass

            return audio_bytes

        finally:
            # Clean up temporary video file
            try:
                # Close video and audio objects first
                if "video" in locals():
                    video.close()
                if "audio" in locals() and audio:
                    audio.close()

                # Wait a bit before trying to delete
                import time

                time.sleep(0.1)

                os.unlink(tmp_file_path)
            except (OSError, PermissionError):
                pass

    except ImportError:
        logger.error(
            "MoviePy not installed. Please install it with: pip install moviepy"
        )
        return None
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        return None


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe audio to text for text sentiment analysis.

    Args:
        audio_bytes: Audio bytes in WAV format

    Returns:
        Transcribed text string
    """
    if audio_bytes is None:
        return ""

    try:
        import tempfile
        import speech_recognition as sr

        # Save audio bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name

        try:
            # Initialize recognizer
            recognizer = sr.Recognizer()

            # Load audio file
            with sr.AudioFile(tmp_file_path) as source:
                # Read audio data
                audio_data = recognizer.record(source)

                # Transcribe using Google Speech Recognition
                try:
                    text = recognizer.recognize_google(audio_data)
                    return text
                except sr.UnknownValueError:
                    logger.warning("Speech could not be understood")
                    return ""
                except sr.RequestError as e:
                    logger.error(
                        f"Could not request results from speech recognition service: {e}"
                    )
                    return ""

        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except (OSError, PermissionError):
                pass

    except ImportError:
        logger.error(
            "SpeechRecognition not installed. Please install it with: pip install SpeechRecognition"
        )
        return ""
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return ""
