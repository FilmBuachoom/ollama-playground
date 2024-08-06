#!/bin/bash

echo "---------------------------------------- Custom process ----------------------------------------"

# Check if Java is installed
if type -p java; then
    echo "Java is already installed."
else
    echo "Java is not installed. Installing Java 21..."

    # Update package list and install wget
    apt-get update && apt-get install -y wget

    # Download and install Java 21
    if wget https://download.oracle.com/java/21/latest/jdk-21_linux-x64_bin.deb; then
        dpkg -i jdk-21_linux-x64_bin.deb
        rm jdk-21_linux-x64_bin.deb
        chown root:root -R /usr/lib/jvm
        echo "Java 21 installation completed."
    else
        echo "Failed to download Java."
        exit 1
    fi
fi

# Set Java environment variables directly (more reliable in Docker)
export JAVA_HOME=/usr/lib/jvm/jdk-21.0.4-oracle-x64
export JVM_PATH=/usr/lib/jvm/jdk-21.0.4-oracle-x64/lib/server/libjvm.so

echo "---------------------------------------- End custom process ------------------------------------"

# args
PORT="${PORT:-9099}"
HOST="${HOST:-0.0.0.0}"

# cd to Pipeline
cd pipelines

# Start Pipelines
uvicorn main:app --host "$HOST" --port "$PORT" --forwarded-allow-ips '*'