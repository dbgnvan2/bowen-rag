#!/bin/bash

# Bowen RAG — Streamlit web app launch script
cd "$(dirname "$0")"

# Start Ollama in background if not already running
if ! pgrep -x ollama > /dev/null; then
    echo "Starting Ollama..."
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 2
fi

echo "Starting Bowen RAG web app at http://localhost:8501"
streamlit run streamlit_app.py
