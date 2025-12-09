# Use full Python 3.11 image (avoids TF issues in slim)
FROM python:3.11

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# Install system dependencies required by TensorFlow and Pillow
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and templates
COPY . .

# Copy your Keras model
COPY best_model_v3.keras ./best_model_v3.keras

# Expose Flask default port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]
