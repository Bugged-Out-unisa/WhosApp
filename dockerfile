# This version of the Dockerfile includes your processed datasets
# Think of this as creating a container that comes with your data pre-installed

FROM ubuntu:22.04

# ... (all the previous setup steps remain the same)
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg \
    software-properties-common \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.12 /usr/bin/python

RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

WORKDIR /app

# Install dependencies first (for layer caching)
COPY requirements.txt ./
COPY package*.json ./
COPY frontend/package*.json ./frontend/

RUN pip install --no-cache-dir -r requirements.txt
RUN npm install

WORKDIR /app/frontend
RUN npm install
WORKDIR /app

# Copy application code
COPY . .

# Here's the key addition: copy your pre-processed datasets
# This ensures your container always has the data it needs
COPY data/datasets/ ./data/datasets/

# If you have trained models, include them too
# COPY models/ ./models/

# Create other necessary directories
RUN mkdir -p logs/aliases logs/dataset logs/pipeline logs/training

# Set up permissions
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app

# Set up startup script
USER root
COPY start.sh /app/
RUN chmod +x /app/start.sh && chown appuser:appuser /app/start.sh

USER appuser

EXPOSE 3000 8000

CMD ["./start.sh"]