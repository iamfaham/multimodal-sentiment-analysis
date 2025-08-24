#!/usr/bin/env python3
"""
Test the updated Google Drive download function
"""

from simple_model_manager import SimpleModelManager


def test_download():
    """Test the download function"""
    print("Testing Google Drive Download Function")
    print("=" * 50)

    # Initialize manager
    manager = SimpleModelManager()

    # Check model status
    status = manager.get_model_status()
    print("Model Status:")
    for model_type, info in status.items():
        print(f"  {model_type}: {'✅' if info['configured'] else '❌'} {info['url']}")
        if info["cached"]:
            print(f"    📁 Cached: {info['filename']}")

    # Test vision model download
    if status["vision"]["configured"]:
        print(f"\nTesting vision model download...")
        try:
            vision_model, device, num_classes = manager.load_vision_model()
            print(f"✅ Vision model loaded: {num_classes} classes")
        except Exception as e:
            print(f"❌ Vision model failed: {e}")
    else:
        print("❌ Vision model not configured")

    # Test audio model download
    if status["audio"]["configured"]:
        print(f"\nTesting audio model download...")
        try:
            audio_model, device = manager.load_audio_model()
            print(f"✅ Audio model loaded")
        except Exception as e:
            print(f"❌ Audio model failed: {e}")
    else:
        print("❌ Audio model not configured")


if __name__ == "__main__":
    test_download()
