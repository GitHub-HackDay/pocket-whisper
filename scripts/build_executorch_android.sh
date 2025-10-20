#!/bin/bash
#
# Build ExecuTorch Android AAR from source
# ExecuTorch is not available on Maven, so we must build it ourselves.
#
# This script:
# 1. Clones the ExecuTorch repository
# 2. Builds the Android AAR
# 3. Copies it to the app/libs/ directory
#

set -e  # Exit on error

echo "========================================================================"
echo "BUILDING EXECUTORCH ANDROID AAR"
echo "========================================================================"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEMP_DIR="/tmp/executorch_build_$$"
AAR_NAME="executorch.aar"

echo ""
echo "Project root: $PROJECT_ROOT"
echo "Temporary build directory: $TEMP_DIR"
echo ""

# Step 1: Clone ExecuTorch
echo "[1/4] Cloning ExecuTorch repository..."
git clone --depth=1 https://github.com/pytorch/executorch.git "$TEMP_DIR"
cd "$TEMP_DIR"
echo "  ✓ ExecuTorch cloned"

# Step 2: Install Python dependencies
echo ""
echo "[2/4] Installing Python dependencies..."
pip install -r requirements.txt
echo "  ✓ Dependencies installed"

# Step 3: Build Android AAR
echo ""
echo "[3/4] Building Android AAR..."
echo "  (This may take 10-20 minutes...)"

# Make sure NDK is available
if [ -z "$ANDROID_NDK" ]; then
    echo "  Warning: ANDROID_NDK not set, script will try to find it"
    # Try to find NDK
    if [ -d "$HOME/Library/Android/sdk/ndk" ]; then
        export ANDROID_NDK="$HOME/Library/Android/sdk/ndk/$(ls $HOME/Library/Android/sdk/ndk | tail -1)"
        echo "  Found NDK: $ANDROID_NDK"
    elif [ -d "$HOME/Android/Sdk/ndk" ]; then
        export ANDROID_NDK="$HOME/Android/Sdk/ndk/$(ls $HOME/Android/Sdk/ndk | tail -1)"
        echo "  Found NDK: $ANDROID_NDK"
    else
        echo "  ERROR: Could not find Android NDK"
        echo "  Please install via Android Studio (Tools -> SDK Manager -> SDK Tools -> NDK)"
        exit 1
    fi
fi

# Build the AAR
./build/build_android.sh

# Find the generated AAR
AAR_PATH=$(find build -name "executorch*.aar" | head -1)

if [ -z "$AAR_PATH" ]; then
    echo "  ERROR: AAR file not found after build"
    exit 1
fi

echo "  ✓ AAR built: $AAR_PATH"

# Step 4: Copy to project
echo ""
echo "[4/4] Copying AAR to project..."
DEST_DIR="$PROJECT_ROOT/app/libs"
mkdir -p "$DEST_DIR"
cp "$AAR_PATH" "$DEST_DIR/$AAR_NAME"

AAR_SIZE=$(du -h "$DEST_DIR/$AAR_NAME" | cut -f1)
echo "  ✓ Copied to: $DEST_DIR/$AAR_NAME"
echo "  ✓ Size: $AAR_SIZE"

# Cleanup
echo ""
echo "Cleaning up temporary files..."
cd "$PROJECT_ROOT"
rm -rf "$TEMP_DIR"
echo "  ✓ Temporary files removed"

echo ""
echo "========================================================================"
echo "✅ EXECUTORCH ANDROID AAR BUILD COMPLETE!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Sync Gradle in Android Studio"
echo "  2. Verify the AAR is recognized in build.gradle.kts"
echo "  3. Test loading a model with ModelLoader.kt"
echo ""

