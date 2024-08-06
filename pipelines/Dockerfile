FROM python:3.11-slim-bookworm

# Expose the port
ENV HOST="0.0.0.0"
ENV PORT="9099"

# Use args
ARG USE_CUDA=true
ARG USE_CUDA_VER=cu121

## Basis ##
ENV ENV=prod \
    PORT=9099 \
    # pass build args to the build
    USE_CUDA_DOCKER=${USE_CUDA} \
    USE_CUDA_DOCKER_VER=${USE_CUDA_VER}


# Install GCC and build tools
RUN apt-get update && \
    apt-get install -y gcc build-essential curl git  wget g++ sed && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the application code
COPY . .

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install uv && \
        if [ "$USE_CUDA" = "true" ]; then \
            pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$USE_CUDA_DOCKER_VER --no-cache-dir; \
        else \
            pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir; \    
        fi
RUN uv pip install --system -r requirements.txt
# RUN uv pip install --system -r requirements.txt --no-cache-dir

# ---------------------------------------- Custom process ----------------------------------------
# Install JDK21
RUN wget https://download.oracle.com/java/21/latest/jdk-21_linux-x64_bin.deb \
    && dpkg -i jdk-21_linux-x64_bin.deb \
    && rm jdk-21_linux-x64_bin.deb \
    && chown root:root -R /usr/lib/jvm

# Install custom python libs
RUN pip install -r requirements_custom.txt

# Removing carriage return characters (\r)
RUN sed -i 's/\r//' start.sh

# Clone Pipelines
RUN rm -rf /pipelines && git clone https://github.com/open-webui/pipelines.git
# ---------------------------------------- End custom process ------------------------------------