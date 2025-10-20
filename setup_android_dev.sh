#!/bin/bash

echo "🚀 Setting up Android Development Environment for Samsung Galaxy S25 Ultra"
echo "=================================================================="

# Set environment variables
export ANDROID_HOME=$HOME/Library/Android/sdk
export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"

echo "✅ Environment variables set:"
echo "   ANDROID_HOME: $ANDROID_HOME"
echo "   JAVA_HOME: $JAVA_HOME"

# Check if device is connected
echo ""
echo "📱 Checking device connection..."
adb devices

echo ""
echo "🔧 Android Development Setup Complete!"
echo ""
echo "Next steps:"
echo "1. Connect your Samsung Galaxy S25 Ultra via USB"
echo "2. Enable Developer Options and USB Debugging on your phone"
echo "3. Run: adb devices (should show your device)"
echo "4. Open Android Studio and import the android-app folder"
echo "5. Build and deploy to your S25 Ultra!"
echo ""
echo "📋 Device Setup Instructions:"
echo "   Settings → About Phone → Software Information → Tap 'Build Number' 7 times"
echo "   Settings → Developer Options → USB Debugging (ON)"
echo "   Settings → Developer Options → Install via USB (ON)"
echo ""
echo "🎯 Your Pocket Whisper app is ready for development!"
