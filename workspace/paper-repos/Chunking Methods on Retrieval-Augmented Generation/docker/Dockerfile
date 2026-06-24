FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential git curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt punkt_tab
RUN python -m spacy download en_core_web_sm

COPY . .

VOLUME ["/workspace/data", "/workspace/results"]

ENV OPENAI_API_KEY=""
ENV OPENAI_API_BASE=""
ENV PYTHONPATH="/workspace/src"

CMD ["python", "run_full_bench.py", "--config", "configs/config.yaml"]
