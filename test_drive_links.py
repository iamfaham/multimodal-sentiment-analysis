#!/usr/bin/env python3
"""
Test Google Drive links directly to see what's being returned
"""

import requests
import os
from pathlib import Path


def test_drive_link(url, filename):
    """Test a Google Drive link directly"""
    print(f"\nTesting: {filename}")
    print(f"URL: {url}")

    try:
        # Convert Google Drive share link to direct download link
        if "drive.google.com" in url:
            if "/file/d/" in url:
                file_id = url.split("/file/d/")[1].split("/")[0]
            elif "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]
            else:
                print("❌ Could not extract file ID")
                return

            direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            print(f"Direct URL: {direct_url}")
        else:
            direct_url = url

        # Test the download
        print("Downloading...")
        response = requests.get(direct_url, stream=True)

        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'Unknown')}")
        print(f"Content-Length: {response.headers.get('content-length', 'Unknown')}")

        if response.status_code == 200:
            # Read first 200 bytes to check content
            content = response.raw.read(200)
            print(f"First 200 bytes: {content[:100]}...")

            # Check if it's HTML
            if content.startswith(b"<!DOCTYPE") or content.startswith(b"<html"):
                print("❌ ERROR: This is an HTML page, not a model file!")
                print("   Your Google Drive link is not working properly")
                print("   Check file permissions and sharing settings")
            else:
                print("✅ Looks like a valid file (not HTML)")

                # Save a small sample to check
                sample_path = f"sample_{filename}"
                with open(sample_path, "wb") as f:
                    f.write(content)
                print(f"Saved sample to: {sample_path}")

        else:
            print(f"❌ Download failed with status: {response.status_code}")

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    print("Google Drive Link Tester")
    print("=" * 50)

    # Check environment variables
    vision_url = os.getenv("VISION_MODEL_DRIVE_LINK")
    audio_url = os.getenv("AUDIO_MODEL_DRIVE_LINK")

    if not vision_url and not audio_url:
        print("❌ No environment variables found!")
        print("Please run setup_env.py first or set:")
        print("  VISION_MODEL_DRIVE_LINK")
        print("  AUDIO_MODEL_DRIVE_LINK")
        return

    if vision_url:
        test_drive_link(vision_url, "resnet50_model.pth")

    if audio_url:
        test_drive_link(audio_url, "wav2vec2_model.pth")

    print("\n" + "=" * 50)
    print("If you see HTML content, your Google Drive links need fixing!")
    print("Make sure:")
    print("  1. Files are set to 'Anyone with the link can view'")
    print("  2. You're using direct file links, not folder links")
    print("  3. Files are not too large for direct download")


if __name__ == "__main__":
    main()
