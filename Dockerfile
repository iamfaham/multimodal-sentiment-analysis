# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV, audio, and moviepy
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    ffmpeg \
    libav-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Set environment variables to prefer pre-built wheels
ENV PIP_PREFER_BINARY=1
ENV PIP_NO_BUILD_ISOLATION=0

# Upgrade pip and install numpy first (to avoid conflicts)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir "numpy>=1.24.0,<2.0.0"

# Install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Verify moviepy installation
RUN python -c "import moviepy; print('MoviePy version:', moviepy.__version__)"

# Copy the app and test script
COPY app.py .
COPY simple_model_manager.py .
COPY test_imports.py .

# Test all imports work correctly
RUN python test_imports.py

# Expose port
EXPOSE 7860

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
