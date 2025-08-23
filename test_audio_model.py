#!/usr/bin/env python3
"""
Test script for the Wav2Vec2 audio sentiment analysis model
"""

import os
import torch
import numpy as np
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import tempfile


def test_audio_model():
    """Test the audio model loading and inference"""

    print("🔊 Testing Wav2Vec2 Audio Sentiment Model")
    print("=" * 50)

    # Check if model file exists
    model_path = "models/wav2vec2_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Audio model file not found at: {model_path}")
        return False

    print(f"✅ Found model file: {model_path}")

    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {device}")

        # Load the model checkpoint to check architecture
        checkpoint = torch.load(model_path, map_location=device)
        print(f"📊 Checkpoint keys: {list(checkpoint.keys())}")

        # Check for classifier weights
        if "classifier.weight" in checkpoint:
            num_classes = checkpoint["classifier.weight"].shape[0]
            print(f"📊 Model has {num_classes} output classes")
        else:
            print("⚠️  Could not determine number of classes from checkpoint")
            num_classes = 3  # Default assumption

        # Initialize model
        print("🔄 Initializing Wav2Vec2 model...")
        model_checkpoint = "facebook/wav2vec2-base"
        model = AutoModelForAudioClassification.from_pretrained(
            model_checkpoint, num_labels=num_classes
        )

        # Load trained weights
        print("🔄 Loading trained weights...")
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()

        print("✅ Model loaded successfully!")

        # Test with dummy audio
        print("🧪 Testing inference with dummy audio...")

        # Create dummy audio (1 second of random noise at 16kHz)
        dummy_audio = np.random.randn(16000).astype(np.float32)

        # Load feature extractor
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

        # Preprocess audio
        inputs = feature_extractor(
            dummy_audio,
            sampling_rate=16000,
            max_length=80000,  # 5 seconds * 16000 Hz
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Move to device
        input_values = inputs.input_values.to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(input_values)
            probabilities = torch.softmax(outputs.logits, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            print(f"🔍 Model output shape: {outputs.logits.shape}")
            print(f"🎯 Predicted class: {predicted.item()}")
            print(f"📊 Confidence: {confidence.item():.3f}")
            print(f"📈 All probabilities: {probabilities.squeeze().cpu().numpy()}")

        # Sentiment mapping
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        predicted_sentiment = sentiment_map.get(
            predicted.item(), f"Class_{predicted.item()}"
        )
        print(f"😊 Predicted sentiment: {predicted_sentiment}")

        print("✅ Audio model test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error testing audio model: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def check_audio_model_file():
    """Check the audio model file details"""

    print("\n🔍 Audio Model File Analysis")
    print("=" * 30)

    model_path = "models/wav2vec2_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    # File size
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"📁 File size: {file_size:.1f} MB")

    try:
        # Load checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)

        print(f"🔑 Checkpoint keys ({len(checkpoint)} total):")
        for key, value in checkpoint.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  - {key}: {type(value)}")

        # Check classifier
        if "classifier.weight" in checkpoint:
            num_classes = checkpoint["classifier.weight"].shape[0]
            print(f"\n🎯 Classifier output classes: {num_classes}")
            print(
                f"📊 Classifier weight shape: {checkpoint['classifier.weight'].shape}"
            )
            if "classifier.bias" in checkpoint:
                print(
                    f"📊 Classifier bias shape: {checkpoint['classifier.bias'].shape}"
                )

        # Check wav2vec2 base model
        if "wav2vec2.feature_extractor.conv_layers.0.conv.weight" in checkpoint:
            print(f"🔊 Wav2Vec2 base model: Present")

    except Exception as e:
        print(f"❌ Error analyzing checkpoint: {str(e)}")


if __name__ == "__main__":
    print("🚀 Starting Wav2Vec2 Audio Model Tests")
    print("=" * 60)

    # Check model file
    check_audio_model_file()

    print("\n" + "=" * 60)

    # Test model loading and inference
    success = test_audio_model()

    if success:
        print("\n🎉 All audio model tests passed!")
    else:
        print("\n💥 Audio model tests failed!")
