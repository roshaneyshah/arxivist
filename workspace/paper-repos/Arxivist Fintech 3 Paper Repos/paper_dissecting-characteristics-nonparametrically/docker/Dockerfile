FROM python:3.10-slim

# System dependencies for scipy/sklearn numerical libraries
RUN apt-get update && apt-get install -y \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Install the dcnp package in editable mode
RUN pip install --no-cache-dir -e src/

# Default: run in-sample estimation with synthetic data
CMD ["python", "scripts/run_insample.py", "--config", "configs/config.yaml", "--use-synthetic"]
