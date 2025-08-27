"""
Fused sentiment analysis model combining text, audio, and vision models.
"""

import logging
from typing import Tuple, Optional, List
from PIL import Image

from .text_model import predict_text_sentiment
from .audio_model import predict_audio_sentiment
from .vision_model import predict_vision_sentiment

logger = logging.getLogger(__name__)


def predict_fused_sentiment(
    text: Optional[str] = None,
    audio_bytes: Optional[bytes] = None,
    image: Optional[Image.Image] = None,
) -> Tuple[str, float]:
    """
    Implement ensemble/fusion logic combining all three models.

    Args:
        text: Input text for text sentiment analysis
        audio_bytes: Audio bytes for audio sentiment analysis
        image: Input image for vision sentiment analysis

    Returns:
        Tuple of (fused_sentiment, overall_confidence)
    """
    results = []

    if text:
        text_sentiment, text_conf = predict_text_sentiment(text)
        results.append(("Text", text_sentiment, text_conf))

    if audio_bytes:
        audio_sentiment, audio_conf = predict_audio_sentiment(audio_bytes)
        results.append(("Audio", audio_sentiment, audio_conf))

    if image:
        vision_sentiment, vision_conf = predict_vision_sentiment(image)
        results.append(("Vision", vision_sentiment, vision_conf))

    if not results:
        return "No inputs provided", 0.0

    # Simple ensemble logic (can be enhanced with more sophisticated fusion strategies)
    sentiment_counts = {}
    total_confidence = 0
    modality_weights = {"Text": 0.3, "Audio": 0.35, "Vision": 0.35}  # Weighted voting

    for modality, sentiment, confidence in results:
        if sentiment not in sentiment_counts:
            sentiment_counts[sentiment] = {"count": 0, "weighted_conf": 0}

        sentiment_counts[sentiment]["count"] += 1
        weight = modality_weights.get(modality, 0.33)
        sentiment_counts[sentiment]["weighted_conf"] += confidence * weight
        total_confidence += confidence

    # Weighted majority voting with confidence averaging
    if sentiment_counts:
        # Find sentiment with highest weighted confidence
        final_sentiment = max(
            sentiment_counts.keys(), key=lambda s: sentiment_counts[s]["weighted_conf"]
        )

        # Calculate overall confidence as weighted average
        avg_confidence = total_confidence / len(results)

        logger.info(
            f"Fused sentiment analysis completed: {final_sentiment} (confidence: {avg_confidence:.2f})"
        )
        logger.info(f"Individual results: {results}")

        return final_sentiment, avg_confidence
    else:
        return "No valid predictions", 0.0


def get_fusion_strategy_info() -> dict:
    """Get information about the fusion strategy."""
    return {
        "strategy_name": "Weighted Ensemble Fusion",
        "description": "Combines predictions from text, audio, and vision models using weighted voting",
        "modality_weights": {"Text": 0.3, "Audio": 0.35, "Vision": 0.35},
        "fusion_method": "Weighted majority voting with confidence averaging",
        "advantages": [
            "Robust to individual model failures",
            "Leverages complementary information from different modalities",
            "Configurable modality weights",
            "Real-time ensemble prediction",
        ],
        "use_cases": [
            "Multi-modal content analysis",
            "Enhanced sentiment accuracy",
            "Cross-validation of predictions",
            "Comprehensive emotional understanding",
        ],
    }


def analyze_modality_agreement(
    text: Optional[str] = None,
    audio_bytes: Optional[bytes] = None,
    image: Optional[Image.Image] = None,
) -> dict:
    """
    Analyze agreement between different modalities.

    Args:
        text: Input text
        audio_bytes: Audio bytes
        image: Input image

    Returns:
        Dictionary containing agreement analysis
    """
    results = {}

    if text:
        text_sentiment, text_conf = predict_text_sentiment(text)
        results["text"] = {"sentiment": text_sentiment, "confidence": text_conf}

    if audio_bytes:
        audio_sentiment, audio_conf = predict_audio_sentiment(audio_bytes)
        results["audio"] = {"sentiment": audio_sentiment, "confidence": audio_conf}

    if image:
        vision_sentiment, vision_conf = predict_vision_sentiment(image)
        results["vision"] = {"sentiment": vision_sentiment, "confidence": vision_conf}

    if len(results) < 2:
        return {"agreement_level": "insufficient_modalities", "details": results}

    # Analyze agreement
    sentiments = [result["sentiment"] for result in results.values()]
    unique_sentiments = set(sentiments)

    if len(unique_sentiments) == 1:
        agreement_level = "perfect"
        agreement_score = 1.0
    elif len(unique_sentiments) == 2:
        agreement_level = "partial"
        agreement_score = 0.5
    else:
        agreement_level = "low"
        agreement_score = 0.0

    # Calculate confidence consistency
    confidences = [result["confidence"] for result in results.values()]
    confidence_std = sum(confidences) / len(confidences) if confidences else 0

    return {
        "agreement_level": agreement_level,
        "agreement_score": agreement_score,
        "modalities_analyzed": len(results),
        "sentiment_distribution": {s: sentiments.count(s) for s in unique_sentiments},
        "confidence_consistency": confidence_std,
        "individual_results": results,
        "recommendation": _get_agreement_recommendation(agreement_level, len(results)),
    }


def _get_agreement_recommendation(agreement_level: str, num_modalities: int) -> str:
    """Get recommendation based on agreement level."""
    if agreement_level == "perfect":
        return "High confidence in prediction - all modalities agree"
    elif agreement_level == "partial":
        return "Moderate confidence - consider modality-specific factors"
    elif agreement_level == "low":
        return "Low confidence - modalities disagree, consider context"
    else:
        return "Insufficient data for reliable fusion"
