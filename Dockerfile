FROM ubuntu:24.04

# Install system deps and CUDA
RUN apt-get update && apt-get install -y \
    python3-pip \
    cuda-12-2 \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA env
ENV LD_LIBRARY_PATH /usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . /app
WORKDIR /app

# Run
CMD ["python", "src/main.py"]
