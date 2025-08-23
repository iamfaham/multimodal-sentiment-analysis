#!/usr/bin/env python3
"""
Startup script for the Sentiment Analysis Testing Ground Streamlit application.
This script provides an easy way to launch the application with proper configuration.
"""

import subprocess
import sys
import os


def main():
    """Main function to start the Streamlit application."""

    print("🧠 Starting Sentiment Analysis Testing Ground...")
    print("=" * 50)

    # Check if app.py exists
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found in current directory!")
        print("Please make sure you're in the correct directory.")
        sys.exit(1)

    # Check if requirements are installed
    try:
        import streamlit
        import pandas
        import PIL

        print("✅ Dependencies check passed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)

    print("🚀 Launching Streamlit application...")
    print("📱 The app will open in your default browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("=" * 50)

    try:
        # Start Streamlit with the app
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.headless",
                "false",
                "--server.port",
                "8501",
            ]
        )
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
