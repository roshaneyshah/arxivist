FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Install project in editable mode
RUN pip install -e .

# Create directories for data, checkpoints, logs
RUN mkdir -p data checkpoints logs results

# Default command: show help
CMD ["python", "train.py", "--help"]
