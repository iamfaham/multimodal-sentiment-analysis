"""
File handling utilities for different input types.
"""

import os
import tempfile
import logging
from typing import Optional, Union, Tuple
from pathlib import Path

from ..config.settings import (
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
)

logger = logging.getLogger(__name__)


def validate_file_format(filename: str, supported_formats: list) -> bool:
    """
    Validate if a file has a supported format.

    Args:
        filename: Name of the file to validate
        supported_formats: List of supported file extensions

    Returns:
        True if file format is supported, False otherwise
    """
    if not filename:
        return False

    file_extension = Path(filename).suffix.lower().lstrip(".")
    return file_extension in supported_formats


def validate_image_file(filename: str) -> bool:
    """Validate if a file is a supported image format."""
    return validate_file_format(filename, SUPPORTED_IMAGE_FORMATS)


def validate_audio_file(filename: str) -> bool:
    """Validate if a file is a supported audio format."""
    return validate_file_format(filename, SUPPORTED_AUDIO_FORMATS)


def validate_video_file(filename: str) -> bool:
    """Validate if a file is a supported video format."""
    return validate_file_format(filename, SUPPORTED_VIDEO_FORMATS)


def get_file_info(file_object) -> dict:
    """
    Extract file information from a file object.

    Args:
        file_object: File object (e.g., StreamlitUploadedFile)

    Returns:
        Dictionary containing file information
    """
    try:
        if hasattr(file_object, "getvalue"):
            file_size = len(file_object.getvalue())
            file_name = getattr(file_object, "name", "Unknown")
        else:
            file_size = len(file_object)
            file_name = "Unknown"

        file_extension = (
            Path(file_name).suffix.lower().lstrip(".")
            if file_name != "Unknown"
            else "Unknown"
        )

        return {
            "name": file_name,
            "size_bytes": file_size,
            "size_kb": file_size / 1024,
            "size_mb": file_size / (1024 * 1024),
            "extension": file_extension,
            "is_valid_image": (
                validate_image_file(file_name) if file_extension != "Unknown" else False
            ),
            "is_valid_audio": (
                validate_audio_file(file_name) if file_extension != "Unknown" else False
            ),
            "is_valid_video": (
                validate_video_file(file_name) if file_extension != "Unknown" else False
            ),
        }
    except Exception as e:
        logger.error(f"Error getting file info: {str(e)}")
        return {
            "name": "Unknown",
            "size_bytes": 0,
            "size_kb": 0,
            "size_mb": 0,
            "extension": "Unknown",
            "is_valid_image": False,
            "is_valid_audio": False,
            "is_valid_video": False,
        }


def create_temp_file(
    suffix: str = "", prefix: str = "temp_"
) -> Tuple[str, tempfile.NamedTemporaryFile]:
    """
    Create a temporary file with proper cleanup handling.

    Args:
        suffix: File extension suffix
        prefix: File name prefix

    Returns:
        Tuple of (file_path, temp_file_object)
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, prefix=prefix, delete=False)
    return temp_file.name, temp_file


def cleanup_temp_file(file_path: str) -> bool:
    """
    Safely cleanup a temporary file.

    Args:
        file_path: Path to the temporary file

    Returns:
        True if cleanup was successful, False otherwise
    """
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            return True
        return True
    except (OSError, PermissionError) as e:
        logger.warning(f"Could not delete temporary file {file_path}: {e}")
        return False


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: File size in bytes

    Returns:
        Formatted file size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def safe_file_operation(operation_func, *args, **kwargs):
    """
    Safely execute a file operation with proper error handling.

    Args:
        operation_func: Function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the operation or None if it fails
    """
    try:
        return operation_func(*args, **kwargs)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
        return None
    except OSError as e:
        logger.error(f"OS error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in file operation: {e}")
        return None
