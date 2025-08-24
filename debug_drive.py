#!/usr/bin/env python3
"""
Debug Google Drive download issues
"""

import os
import requests
import re
from pathlib import Path


# Load .env file manually
def load_env():
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip().strip('"')


def test_drive_bypass(file_id):
    """Test different bypass methods"""
    print(f"Testing file ID: {file_id}")
    print("=" * 50)

    # Method 1: Direct bypass
    print("\n1. Testing direct bypass...")
    try:
        url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        response = requests.get(url, stream=True)
        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'Unknown')}")

        first_chunk = next(response.iter_content(chunk_size=1024), b"")
        if first_chunk.startswith(b"<!DOCTYPE") or first_chunk.startswith(b"<html"):
            print("❌ Still getting HTML")
            html_content = first_chunk.decode("utf-8", errors="ignore")
            print(f"HTML preview: {html_content[:200]}...")
        else:
            print("✅ Got file content!")
            print(f"First bytes: {first_chunk[:50]}")
            return True
    except Exception as e:
        print(f"❌ Error: {e}")

    # Method 2: Session-based approach
    print("\n2. Testing session-based approach...")
    try:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
        )

        # First get the virus scan page
        virus_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = session.get(virus_url)
        print(f"Virus page status: {response.status_code}")

        # Extract confirm and UUID
        html_content = response.text
        confirm_match = re.search(r'name="confirm" value="([^"]+)"', html_content)
        uuid_match = re.search(r'name="uuid" value="([^"]+)"', html_content)

        if confirm_match and uuid_match:
            confirm_value = confirm_match.group(1)
            uuid_value = uuid_match.group(1)
            print(f"Found confirm: {confirm_value}")
            print(f"Found UUID: {uuid_value}")

            # Submit form
            form_data = {
                "id": file_id,
                "export": "download",
                "confirm": confirm_value,
                "uuid": uuid_value,
            }
            form_url = "https://drive.usercontent.google.com/download"
            response = session.post(form_url, data=form_data, stream=True)

            print(f"Form submission status: {response.status_code}")
            first_chunk = next(response.iter_content(chunk_size=1024), b"")

            if first_chunk.startswith(b"<!DOCTYPE") or first_chunk.startswith(b"<html"):
                print("❌ Form submission still returned HTML")
            else:
                print("✅ Form submission successful!")
                return True
        else:
            print("❌ Could not extract confirm/UUID")
            print(f"HTML preview: {html_content[:300]}...")

    except Exception as e:
        print(f"❌ Error: {e}")

    # Method 3: Extract download URL from file page
    print("\n3. Testing file page extraction...")
    try:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://drive.google.com/",
            }
        )

        file_url = f"https://drive.google.com/file/d/{file_id}/view"
        response = session.get(file_url)
        print(f"File page status: {response.status_code}")

        if response.status_code == 200:
            # Look for download URL in the page
            download_match = re.search(r'"downloadUrl":"([^"]+)"', response.text)
            if download_match:
                download_url = (
                    download_match.group(1)
                    .replace("\\u003d", "=")
                    .replace("\\u0026", "&")
                )
                print(f"Found download URL: {download_url}")

                # Try downloading from this URL
                response = session.get(download_url, stream=True)
                first_chunk = next(response.iter_content(chunk_size=1024), b"")

                if first_chunk.startswith(b"<!DOCTYPE") or first_chunk.startswith(
                    b"<html"
                ):
                    print("❌ Download URL still returned HTML")
                else:
                    print("✅ Download URL successful!")
                    return True
            else:
                print("❌ Could not find download URL in page")
        else:
            print(f"❌ Could not access file page")

    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n❌ All methods failed")
    return False


def main():
    print("Google Drive Bypass Debug Tool")
    print("=" * 50)

    # Load environment variables
    load_env()

    # Get file ID from environment or user input
    vision_url = os.getenv("VISION_MODEL_DRIVE_LINK", "")
    audio_url = os.getenv("AUDIO_MODEL_DRIVE_LINK", "")

    if not vision_url and not audio_url:
        print("❌ No environment variables found!")
        print("Please set VISION_MODEL_DRIVE_LINK or AUDIO_MODEL_DRIVE_LINK")
        return

    if vision_url:
        print(f"\nTesting Vision Model URL: {vision_url}")
        if "/file/d/" in vision_url:
            file_id = vision_url.split("/file/d/")[1].split("/")[0]
            test_drive_bypass(file_id)
        else:
            print("❌ Invalid vision model URL format")

    if audio_url:
        print(f"\nTesting Audio Model URL: {audio_url}")
        if "/file/d/" in audio_url:
            file_id = audio_url.split("/file/d/")[1].split("/")[0]
            test_drive_bypass(file_id)
        else:
            print("❌ Invalid audio model URL format")


if __name__ == "__main__":
    main()
