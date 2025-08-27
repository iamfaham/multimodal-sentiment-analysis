"""
Text sentiment analysis model using TextBlob.
"""

import logging
from typing import Tuple, Optional
from ..config.settings import TEXT_MODEL_CONFIG

logger = logging.getLogger(__name__)


def predict_text_sentiment(text: str) -> Tuple[str, float]:
    """
    Analyze text sentiment using TextBlob.

    Args:
        text: Input text to analyze

    Returns:
        Tuple of (sentiment, confidence)
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
        confidence_threshold = TEXT_MODEL_CONFIG["confidence_threshold"]

        if polarity > confidence_threshold:
            sentiment = "Positive"
            confidence = min(0.95, 0.6 + abs(polarity) * 0.3)
        elif polarity < -confidence_threshold:
            sentiment = "Negative"
            confidence = min(0.95, 0.6 + abs(polarity) * 0.3)
        else:
            sentiment = "Neutral"
            confidence = 0.7 - abs(polarity) * 0.2

        # Round confidence to 2 decimal places
        confidence = round(confidence, 2)

        logger.info(
            f"Text sentiment analysis completed: {sentiment} (confidence: {confidence})"
        )
        return sentiment, confidence

    except ImportError:
        logger.error(
            "TextBlob not installed. Please install it with: pip install textblob"
        )
        return "TextBlob not available", 0.0
    except Exception as e:
        logger.error(f"Error in text sentiment analysis: {str(e)}")
        return "Error occurred", 0.0


def get_text_model_info() -> dict:
    """Get information about the text sentiment model."""
    return {
        "model_name": TEXT_MODEL_CONFIG["model_name"],
        "description": "Natural Language Processing based sentiment analysis using TextBlob",
        "capabilities": [
            "Text sentiment classification (Positive/Negative/Neutral)",
            "Confidence scoring",
            "Real-time analysis",
            "No external API required",
        ],
        "input_format": "Plain text",
        "output_format": "Sentiment label + confidence score",
    }
