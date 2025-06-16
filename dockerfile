# This version of the Dockerfile includes your processed datasets
# Think of this as creating a container that comes with your data pre-installed

FROM ubuntu:22.04

# Prevent interactive prompts during package installation
# This is like telling the installer "just use default settings for everything"
ENV DEBIAN_FRONTEND=noninteractive

# Install basic system dependencies first
# We're building the foundation before adding specialized tools
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg \
    software-properties-common \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Add the deadsnakes PPA to get access to newer Python versions
# This is like adding a specialty store to your shopping list
# The deadsnakes PPA is a trusted repository that provides newer Python versions for Ubuntu
RUN add-apt-repository ppa:deadsnakes/ppa -y && apt-get update

# Now install Python 3.12 and related packages
# Since we added the PPA, these packages should now be available
# Note: python3.12-distutils is not available because distutils is deprecated in Python 3.12
# Modern Python applications use setuptools instead, which we'll install via pip
RUN apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Install pip separately using get-pip.py method
# This is more reliable than trying to install python3-pip for newer Python versions
# The get-pip.py script automatically includes setuptools, which replaces the deprecated distutils
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py

# Verify that setuptools is available (this replaces distutils functionality)
# We also install wheel, which is essential for building many Python packages
RUN python3.12 -m pip install --upgrade setuptools wheel

# Create a symbolic link so 'python' command works
# This makes it easier to run Python commands without specifying the version
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# Install Node.js 18.x
# This adds the Node.js repository and installs Node.js for your frontend
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Set up the working directory
WORKDIR /app

# Copy and install Python dependencies first
# We do this early to take advantage of Docker's layer caching
# If your requirements.txt doesn't change, Docker can reuse this step
COPY requirements.txt ./

# Install dependencies with verbose output to help diagnose any issues
# The -v flag gives us detailed information about what pip is doing
# We'll also upgrade pip first to ensure we have the latest version
RUN python -m pip install --upgrade pip

# After installing Python, create a virtual environment
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

RUN python -m pip install --no-cache-dir --force-reinstall -v -r requirements.txt

# Copy and install Node.js dependencies
COPY package*.json ./
RUN npm install

# Install frontend dependencies
COPY frontend/package*.json ./frontend/
WORKDIR /app/frontend
RUN npm install
WORKDIR /app

# Copy your application code
# We do this after installing dependencies so code changes don't force dependency reinstalls
COPY . .

# Copy your pre-processed datasets
# This ensures your container always has the data it needs
COPY data/datasets/ ./data/datasets/

# If you have trained models, include them too
# Uncomment the next line if you have pre-trained models to include
# COPY models/ ./models/

# Create necessary directories for logs and temporary files
RUN mkdir -p logs/aliases logs/dataset logs/pipeline logs/training

# Set up user permissions for security
# Running as root inside containers is a security risk, so we create a regular user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app

# Set up the startup script
USER root
COPY start.sh /app/
RUN chmod +x /app/start.sh && chown appuser:appuser /app/start.sh

# Switch to the application user for running the application
USER appuser

# Expose the ports your application uses
EXPOSE 3000 8000

# Define the default command to run when the container starts
CMD ["./start.sh"]