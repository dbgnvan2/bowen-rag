#!/bin/bash

# Bowen RAG — launch script
cd "$(dirname "$0")"

# Start Ollama in background if not already running
if ! pgrep -x ollama > /dev/null; then
    echo "Starting Ollama..."
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 2
fi

python3 bowen_rag_gui.py
