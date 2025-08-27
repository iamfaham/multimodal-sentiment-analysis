"""
UI styles and CSS for the Sentiment Fused application.
"""

from ..config.settings import CUSTOM_CSS, UI_COLORS


def get_custom_css() -> str:
    """Get the custom CSS styles for the application."""
    return CUSTOM_CSS


def get_ui_colors() -> dict:
    """Get the UI color scheme."""
    return UI_COLORS


def get_sentiment_color_style(sentiment: str) -> str:
    """Get color style for different sentiment types."""
    colors = {
        "Positive": "color: #28a745;",
        "Negative": "color: #dc3545;",
        "Neutral": "color: #ffc107;",
        "Angry": "color: #dc3545;",
        "Sad": "color: #17a2b8;",
        "Happy": "color: #28a745;",
        "Fear": "color: #6f42c1;",
        "Disgust": "color: #fd7e14;",
        "Surprise": "color: #ffc107;",
    }
    return colors.get(sentiment, "color: #6c757d;")


def get_metric_style(metric_type: str = "default") -> str:
    """Get styling for different metric types."""
    styles = {
        "default": "background-color: #f8f9fa; padding: 1rem; border-radius: 8px;",
        "success": "background-color: #d4edda; padding: 1rem; border-radius: 8px; border: 1px solid #c3e6cb;",
        "warning": "background-color: #fff3cd; padding: 1rem; border-radius: 8px; border: 1px solid #ffeaa7;",
        "error": "background-color: #f8d7da; padding: 1rem; border-radius: 8px; border: 1px solid #f5c6cb;",
        "info": "background-color: #d1ecf1; padding: 1rem; border-radius: 8px; border: 1px solid #bee5eb;",
    }
    return styles.get(metric_type, styles["default"])


def get_card_style(card_type: str = "default") -> str:
    """Get styling for different card types."""
    styles = {
        "default": "background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #1f77b4;",
        "model": "background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #1f77b4;",
        "result": "background-color: #e8f4fd; padding: 1rem; border-radius: 8px; border: 1px solid #1f77b4; margin: 1rem 0;",
        "upload": "background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border: 2px dashed #dee2e6; text-align: center; margin: 1rem 0;",
        "info": "background-color: #d1ecf1; padding: 1rem; border-radius: 8px; border: 1px solid #bee5eb; margin: 1rem 0;",
    }
    return styles.get(card_type, styles["default"])


def get_button_style(button_type: str = "primary") -> str:
    """Get styling for different button types."""
    styles = {
        "primary": "background-color: #1f77b4; color: white; border: none; padding: 0.5rem 1rem; border-radius: 5px;",
        "secondary": "background-color: #6c757d; color: white; border: none; padding: 0.5rem 1rem; border-radius: 5px;",
        "success": "background-color: #28a745; color: white; border: none; padding: 0.5rem 1rem; border-radius: 5px;",
        "warning": "background-color: #ffc107; color: black; border: none; padding: 0.5rem 1rem; border-radius: 5px;",
        "danger": "background-color: #dc3545; color: white; border: none; padding: 0.5rem 1rem; border-radius: 5px;",
    }
    return styles.get(button_type, styles["primary"])


def get_sidebar_style() -> str:
    """Get styling for the sidebar."""
    return """
    <style>
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    .css-1d391kg .sidebar-content {
        padding: 1rem;
    }
    </style>
    """


def get_header_style() -> str:
    """Get styling for the main header."""
    return """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """
