#!/bin/bash
# Quick setup script for Android Studio

set -e

echo "========================================"
echo "Pocket Whisper - Android Studio Setup"
echo "========================================"
echo ""

# Check if Android Studio is installed
if [ -d "/Applications/Android Studio.app" ]; then
    echo "✅ Android Studio is installed"
else
    echo "❌ Android Studio NOT found in /Applications/"
    echo ""
    echo "Please install Android Studio first:"
    echo "  1. Visit: https://developer.android.com/studio"
    echo "  2. Download and install Android Studio"
    echo "  3. Run the setup wizard (choose Standard installation)"
    echo "  4. Come back and run this script again"
    echo ""
    exit 1
fi

echo ""

# Check for SDK
SDK_LOCATIONS=(
    "$HOME/Library/Android/sdk"
    "$ANDROID_HOME"
    "$ANDROID_SDK_ROOT"
)

SDK_PATH=""
for loc in "${SDK_LOCATIONS[@]}"; do
    if [ -d "$loc" ]; then
        SDK_PATH="$loc"
        echo "✅ Android SDK found at: $SDK_PATH"
        break
    fi
done

if [ -z "$SDK_PATH" ]; then
    echo "❌ Android SDK not found"
    echo ""
    echo "To find your SDK path:"
    echo "  1. Open Android Studio"
    echo "  2. Go to: Preferences → Android SDK"
    echo "  3. Copy the 'Android SDK Location'"
    echo "  4. Run: echo 'sdk.dir=YOUR_SDK_PATH' > local.properties"
    echo ""
    
    # Offer to use default path
    read -p "Would you like to use the default path ($HOME/Library/Android/sdk)? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        SDK_PATH="$HOME/Library/Android/sdk"
        echo "Using: $SDK_PATH"
    else
        exit 1
    fi
fi

echo ""

# Create local.properties
if [ -f "local.properties" ]; then
    echo "⚠️  local.properties already exists"
    read -p "Overwrite it? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing local.properties"
    else
        echo "sdk.dir=$SDK_PATH" > local.properties
        echo "✅ Created local.properties with SDK path"
    fi
else
    echo "sdk.dir=$SDK_PATH" > local.properties
    echo "✅ Created local.properties with SDK path"
fi

echo ""

# Check ADB
if command -v adb &> /dev/null; then
    echo "✅ ADB is available"
    echo ""
    echo "Checking for connected devices..."
    adb devices
    echo ""
else
    echo "⚠️  ADB not found in PATH"
    echo "   Add to your ~/.zshrc or ~/.bash_profile:"
    echo "   export PATH=\"$SDK_PATH/platform-tools:\$PATH\""
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Open the project in Android Studio:"
echo "     open -a 'Android Studio' ."
echo ""
echo "  2. Wait for Gradle sync (5-15 minutes first time)"
echo ""
echo "  3. Connect your S25 Ultra via USB"
echo "     - Enable USB debugging in Developer Options"
echo "     - Allow USB debugging popup on phone"
echo ""
echo "  4. Click the green Play button (▶️) to build & run"
echo ""
echo "See ANDROID_STUDIO_SETUP.md for detailed instructions."
echo ""

