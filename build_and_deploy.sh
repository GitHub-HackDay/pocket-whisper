#!/bin/bash
# Quick build and deploy script for Pocket Whisper

set -e

echo "======================================"
echo "Pocket Whisper - Build & Deploy"
echo "======================================"
echo ""

# Navigate to project directory
cd "$(dirname "$0")"

# Build APK
echo "📦 Building APK..."
./gradlew assembleDebug --no-daemon

# Check if device is connected
echo ""
echo "📱 Checking device connection..."
adb devices | grep -q "device$" || {
    echo "❌ No device connected!"
    echo "Run: cd ../android-dev && ./connect_wireless.sh"
    exit 1
}

# Install APK
echo ""
echo "🚀 Installing on S25 Ultra..."
adb install -r app/build/outputs/apk/debug/app-debug.apk

# Launch app
echo ""
echo "▶️  Launching app..."
adb shell am start -n com.pocketwhisper.app/.MainActivity

# Show logs
echo ""
echo "📋 App logs (Ctrl+C to stop):"
echo "======================================"
adb logcat -s PocketWhisper ModelLoader VadDetector MainActivity:D


