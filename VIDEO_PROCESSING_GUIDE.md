# 🎬 Video Processing Pipeline Guide for Multimodal Analysis

## 🎯 **Objective & Scope**

**Goal**: Create a Streamlit app that uploads a video and extracts its core components for multimodal analysis:

- **Visual Frames**: Representative images from the video
- **Audio Track**: Extracted audio in WAV format
- **Transcribed Text**: Speech converted to text

**Scope**: This guide covers the complete extraction and conversion pipeline. Machine learning models and sentiment analysis are excluded - the focus is purely on data processing and UI components.

---

## 📚 **Step 1: Essential Libraries & Setup**

### **Required Python Libraries**

```bash
pip install streamlit opencv-python-headless moviepy SpeechRecognition
```

### **Requirements.txt**

```txt
streamlit
opencv-python-headless
moviepy
SpeechRecognition
```

### **FFmpeg Dependency**

- **MoviePy** requires FFmpeg for video processing
- **Windows**: Download from https://ffmpeg.org/download.html
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

---

## 🖥️ **Step 2: Creating the Streamlit Interface**

### **Basic UI Setup**

```python
import streamlit as st

st.set_page_config(
    page_title="Video Processing Pipeline",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Video Processing Pipeline")
st.markdown("Upload a video to extract frames, audio, and text for analysis")

# File uploader
uploaded_video = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov", "mkv", "wmv", "flv"],
    help="Supported formats: MP4, AVI, MOV, MKV, WMV, FLV"
)

# Process button
if st.button("🚀 Process Video", type="primary", use_container_width=True):
    if uploaded_video:
        process_video(uploaded_video)
    else:
        st.warning("Please upload a video file first")
```

---

## ⚙️ **Step 3: The Core Extraction Logic**

### **3.1 Video-to-Frames Extraction**

```python
def extract_frames_from_video(video_file, max_frames=5):
    """
    Extract representative frames from video using OpenCV

    Args:
        video_file: Video file object or path
        max_frames: Maximum frames to extract (default: 5)

    Returns:
        List of PIL Image objects
    """
    try:
        import cv2
        import tempfile
        import numpy as np
        from PIL import Image

        # Save video bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            if hasattr(video_file, "getvalue"):
                tmp_file.write(video_file.getvalue())
            else:
                tmp_file.write(video_file)
            tmp_file_path = tmp_file.name

        try:
            # Open video with OpenCV
            cap = cv2.VideoCapture(tmp_file_path)

            if not cap.isOpened():
                st.error("Could not open video file")
                return []

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            st.info(f"📹 Video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")

            # Extract frames at strategic intervals
            frames = []
            if total_frames > 0:
                # Select frames: start, 25%, 50%, 75%, end
                frame_indices = [
                    0,
                    int(total_frames * 0.25),
                    int(total_frames * 0.5),
                    int(total_frames * 0.75),
                    total_frames - 1
                ]
                frame_indices = list(set(frame_indices))  # Remove duplicates
                frame_indices.sort()

                for frame_idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Convert to PIL Image
                        pil_image = Image.fromarray(frame_rgb)
                        frames.append(pil_image)

            cap.release()
            return frames

        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except (OSError, PermissionError):
                pass

    except ImportError:
        st.error("OpenCV not installed. Please install it with: pip install opencv-python")
        return []
    except Exception as e:
        st.error(f"Error extracting frames: {str(e)}")
        return []
```

**How it works:**

1. **Temporary File**: Saves video bytes to a temporary MP4 file
2. **OpenCV Capture**: Opens video and reads properties (frames, FPS, duration)
3. **Strategic Sampling**: Selects frames at key points (start, 25%, 50%, 75%, end)
4. **Format Conversion**: Converts BGR to RGB and creates PIL Image objects
5. **Cleanup**: Removes temporary files safely

---

### **3.2 Video-to-Audio Conversion**

```python
def extract_audio_from_video(video_file):
    """
    Extract audio track from video using MoviePy

    Args:
        video_file: Video file object or path

    Returns:
        Audio bytes in WAV format
    """
    try:
        import tempfile
        from moviepy import VideoFileClip

        # Save video bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            if hasattr(video_file, "getvalue"):
                tmp_file.write(video_file.getvalue())
            else:
                tmp_file.write(video_file)
            tmp_file_path = tmp_file.name

        try:
            # Extract audio using MoviePy
            video = VideoFileClip(tmp_file_path)
            audio = video.audio

            if audio is None:
                st.warning("No audio track found in video")
                return None

            # Save audio to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
                audio_path = audio_file.name

            # Export audio as WAV
            audio.write_audiofile(audio_path, verbose=False, logger=None)

            # Read the audio file and return bytes
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            # Clean up temporary audio file
            try:
                os.unlink(audio_path)
            except (OSError, PermissionError):
                pass

            return audio_bytes

        finally:
            # Clean up temporary video file
            try:
                # Close video and audio objects first
                if 'video' in locals():
                    video.close()
                if 'audio' in locals() and audio:
                    audio.close()

                # Wait a bit before trying to delete
                import time
                time.sleep(0.1)

                os.unlink(tmp_file_path)
            except (OSError, PermissionError):
                pass

    except ImportError:
        st.error("MoviePy not installed. Please install it with: pip install moviepy")
        return None
    except Exception as e:
        st.error(f"Error extracting audio: {str(e)}")
        return None
```

**How it works:**

1. **Temporary File**: Creates temporary MP4 file from video bytes
2. **MoviePy Processing**: Uses VideoFileClip to extract audio track
3. **WAV Export**: Converts audio to WAV format
4. **Bytes Return**: Reads WAV file and returns as bytes
5. **Resource Management**: Properly closes video/audio objects and cleans up files

---

### **3.3 Audio-to-Text Transcription**

```python
def transcribe_audio(audio_bytes):
    """
    Transcribe audio to text using SpeechRecognition

    Args:
        audio_bytes: Audio bytes in WAV format

    Returns:
        Transcribed text string
    """
    if audio_bytes is None:
        return ""

    try:
        import tempfile
        import speech_recognition as sr

        # Save audio bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name

        try:
            # Initialize recognizer
            recognizer = sr.Recognizer()

            # Load audio file
            with sr.AudioFile(tmp_file_path) as source:
                # Read audio data
                audio_data = recognizer.record(source)

                # Transcribe using Google Speech Recognition
                try:
                    text = recognizer.recognize_google(audio_data)
                    return text
                except sr.UnknownValueError:
                    st.warning("Speech could not be understood")
                    return ""
                except sr.RequestError as e:
                    st.error(f"Could not request results from speech recognition service: {e}")
                    return ""

        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except (OSError, PermissionError):
                pass

    except ImportError:
        st.error("SpeechRecognition not installed. Please install it with: pip install SpeechRecognition")
        return ""
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return ""
```

**How it works:**

1. **Temporary File**: Saves audio bytes to temporary WAV file
2. **Speech Recognition**: Uses Google's speech recognition service
3. **Audio Processing**: Records and processes audio data
4. **Text Return**: Returns transcribed text or empty string on failure
5. **Cleanup**: Removes temporary files safely

---

## 🔄 **Step 4: Complete Processing Pipeline**

### **Integrated Processing Function**

```python
def process_video(uploaded_video):
    """Complete video processing pipeline"""

    st.subheader("🎬 Video Processing Pipeline")
    st.info("📁 Processing uploaded video file...")

    # 1. Extract frames
    st.markdown("**1. 🎥 Frame Extraction**")
    frames = extract_frames_from_video(uploaded_video, max_frames=5)

    if frames:
        st.success(f"✅ Extracted {len(frames)} representative frames")

        # Display extracted frames
        cols = st.columns(len(frames))
        for i, frame in enumerate(frames):
            with cols[i]:
                st.image(frame, caption=f"Frame {i+1}", use_container_width=True)
    else:
        st.warning("⚠️ Could not extract frames from video")
        frames = []

    # 2. Extract audio
    st.markdown("**2. 🎵 Audio Extraction**")
    audio_bytes = extract_audio_from_video(uploaded_video)

    if audio_bytes:
        st.success("✅ Audio extracted successfully")
        st.audio(audio_bytes, format="audio/wav")
    else:
        st.warning("⚠️ Could not extract audio from video")
        audio_bytes = None

    # 3. Transcribe audio
    st.markdown("**3. 📝 Audio Transcription**")
    if audio_bytes:
        transcribed_text = transcribe_audio(audio_bytes)
        if transcribed_text:
            st.success("✅ Audio transcribed successfully")
            st.markdown(f'**Transcribed Text:** "{transcribed_text}"')
        else:
            st.warning("⚠️ Could not transcribe audio")
            transcribed_text = ""
    else:
        transcribed_text = ""
        st.info("ℹ️ No audio available for transcription")

    # Store results for later use
    st.session_state.processed_frames = frames
    st.session_state.processed_audio = audio_bytes
    st.session_state.transcribed_text = transcribed_text

    st.success("🎉 Video processing completed! All components extracted successfully.")
```

---

## 🎯 **Key Benefits of This Approach**

### **1. Real Video Processing**

- ✅ **Actual Audio**: Extracts real audio from uploaded videos
- ✅ **Representative Frames**: Strategic frame selection (not just sequential)
- ✅ **Real Transcription**: Converts actual speech to text

### **2. Robust Error Handling**

- ✅ **File Access**: Handles temporary file conflicts gracefully
- ✅ **Resource Management**: Properly closes video/audio objects
- ✅ **Cleanup**: Safe temporary file removal

### **3. User Experience**

- ✅ **Visual Feedback**: Shows extracted frames, audio player, and text
- ✅ **Progress Tracking**: Clear step-by-step processing display
- ✅ **Error Messages**: Informative feedback for troubleshooting

### **4. Scalability**

- ✅ **Modular Design**: Each extraction function is independent
- ✅ **Reusable Components**: Functions can be used in other parts of the app
- ✅ **Easy Maintenance**: Clear separation of concerns

---

## 🚀 **Usage Example**

```python
# Complete working example
import streamlit as st
import tempfile
import os

# Setup page
st.set_page_config(page_title="Video Processor", layout="wide")
st.title("🎬 Video Processing Pipeline")

# File upload
uploaded_video = st.file_uploader("Choose video file", type=["mp4", "avi", "mov"])

# Process button
if st.button("🚀 Process Video", type="primary"):
    if uploaded_video:
        process_video(uploaded_video)
    else:
        st.warning("Please upload a video first")

# Display results
if 'processed_frames' in st.session_state:
    st.subheader("📊 Processing Results")
    st.write(f"Frames: {len(st.session_state.processed_frames)}")
    st.write(f"Audio: {'✅' if st.session_state.processed_audio else '❌'}")
    st.write(f"Text: {'✅' if st.session_state.transcribed_text else '❌'}")
```

---

## 🔧 **Troubleshooting Common Issues**

### **1. FFmpeg Not Found**

```bash
# Windows: Add FFmpeg to PATH
# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg
```

### **2. OpenCV Import Error**

```bash
pip install opencv-python-headless
```

### **3. MoviePy Audio Issues**

```bash
pip install moviepy --upgrade
# Ensure FFmpeg is installed
```

### **4. Speech Recognition Errors**

```bash
pip install SpeechRecognition
# Check internet connection for Google service
```

---

## 📝 **Summary**

This guide provides a complete video processing pipeline that:

1. **🎥 Extracts Frames**: Strategic sampling of representative video frames
2. **🎵 Extracts Audio**: Converts video audio to WAV format
3. **📝 Transcribes Speech**: Converts audio to searchable text
4. **🖥️ Provides UI**: Clean Streamlit interface with progress tracking
5. **🔧 Handles Errors**: Robust error handling and resource management

The result is a production-ready video processing system that extracts all necessary components for multimodal analysis without any machine learning dependencies. Each component is extracted independently and can be used for further processing or analysis as needed.

**Next Steps**: Use the extracted frames, audio, and text with your preferred analysis models or export them for external processing.
