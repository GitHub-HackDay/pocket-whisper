#!/bin/bash
# Quick start script for Pocket Whisper development

echo "🎤 Starting Pocket Whisper Development Environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create Android project if it doesn't exist
if [ ! -d "android_app" ]; then
    echo "📱 Creating Android project..."
    python3 scripts/create_android_project.py
fi

# Convert models if they don't exist
if [ ! -f "models/asr/whisper_tiny.pte" ] && [ ! -f "models/asr/whisper_tiny.pth" ]; then
    echo "🤖 Converting ASR model..."
    python3 scripts/model_converter.py --action convert_asr
fi

if [ ! -f "models/completion/word_predictor.pte" ] && [ ! -f "models/completion/word_predictor.pth" ]; then
    echo "🧠 Converting completion model..."
    python3 scripts/model_converter.py --action convert_completion
fi

# Start development server
echo "🌐 Starting development server..."
python3 scripts/dev_server.py
