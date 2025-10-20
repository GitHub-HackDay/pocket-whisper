@echo off
REM Quick start script for Pocket Whisper development

echo 🎤 Starting Pocket Whisper Development Environment...

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Create Android project if it doesn't exist
if not exist "android_app" (
    echo 📱 Creating Android project...
    python scripts/create_android_project.py
)

REM Convert models if they don't exist
if not exist "models\asr\whisper_tiny.pte" (
    echo 🤖 Converting ASR model...
    python scripts/model_converter.py --action convert_asr
)

if not exist "models\completion\word_predictor.pte" (
    echo 🧠 Converting completion model...
    python scripts/model_converter.py --action convert_completion
)

REM Start development server
echo 🌐 Starting development server...
python scripts/dev_server.py
