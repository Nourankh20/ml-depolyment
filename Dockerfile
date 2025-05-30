# Use Python 3.9 for TensorFlow 2.17.0 compatibility
FROM python:3.9

# Set working directory
WORKDIR /app

# Install system dependencies for TensorFlow
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY model.tflite .

# Expose port for Render
EXPOSE 5000

# Run the Flask app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
