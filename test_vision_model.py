#!/usr/bin/env python3
"""
Test script for the vision sentiment analysis model.
This script verifies that the ResNet-50 model can be loaded and run inference.
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np


def get_sentiment_mapping(num_classes):
    """Get the sentiment mapping based on number of classes"""
    if num_classes == 3:
        return {0: "Negative", 1: "Neutral", 2: "Positive"}
    elif num_classes == 4:
        # Common 4-class emotion mapping
        return {0: "Angry", 1: "Sad", 2: "Happy", 3: "Neutral"}
    elif num_classes == 7:
        # FER2013 7-class emotion mapping
        return {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
    else:
        # Generic mapping for unknown number of classes
        return {i: f"Class_{i}" for i in range(num_classes)}


def test_vision_model():
    """Test the vision sentiment analysis model"""

    print("🧠 Testing Vision Sentiment Analysis Model")
    print("=" * 50)

    # Check if model file exists
    model_path = "models/resnet50_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure the model file exists in the models/ directory")
        return False

    print(f"✅ Model file found: {model_path}")

    try:
        # Load the model weights first to check the architecture
        print("📥 Loading model checkpoint...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check the number of classes from the checkpoint
        if 'fc.weight' in checkpoint:
            num_classes = checkpoint['fc.weight'].shape[0]
            print(f"📊 Model checkpoint has {num_classes} output classes")
        else:
            # Fallback: try to infer from the last layer
            num_classes = 3  # Default assumption
            print("⚠️ Could not determine number of classes from checkpoint, assuming 3")
        
        # Initialize ResNet-50 model with the correct number of classes
        print("🔧 Initializing ResNet-50 model...")
        model = models.resnet50(weights=None)  # Don't load ImageNet weights
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)  # Use actual number of classes
        
        print(f"📥 Loading trained weights for {num_classes} classes...")
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully with {num_classes} classes!")
        print(f"🖥️  Using device: {device}")

        # Test with a dummy image
        print("🧪 Testing inference with dummy image...")

        # Create a dummy image (224x224 RGB)
        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )

        # Apply transforms
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        image_tensor = transform(dummy_image).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            print(f"🔍 Model output shape: {outputs.shape}")
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            # Get sentiment mapping based on number of classes
            sentiment_map = get_sentiment_mapping(num_classes)
            sentiment = sentiment_map[predicted.item()]
            confidence_score = confidence.item()

        print(f"🎯 Test prediction: {sentiment} (confidence: {confidence_score:.3f})")
        print(f"📋 Available classes: {list(sentiment_map.values())}")
        print("✅ Inference test passed!")

        return True

    except Exception as e:
        print(f"❌ Error testing model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    success = test_vision_model()

    if success:
        print("\n🎉 All tests passed! The vision model is ready to use.")
        print("You can now run the Streamlit app with: streamlit run app.py")
    else:
        print("\n💥 Tests failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
