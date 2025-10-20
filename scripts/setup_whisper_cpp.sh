#!/bin/bash
#
# Setup whisper.cpp for Android integration
#
# This script:
# 1. Clones whisper.cpp repository
# 2. Downloads and converts Distil-Whisper model
# 3. Builds Android library (AAR)
# 4. Copies to Android project

set -e  # Exit on error

echo "=========================================="
echo "SETTING UP WHISPER.CPP FOR ANDROID"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WHISPER_DIR="$PROJECT_ROOT/whisper.cpp"
MODELS_DIR="$PROJECT_ROOT/app/src/main/assets"

# Step 1: Clone whisper.cpp if not exists
if [ ! -d "$WHISPER_DIR" ]; then
    echo ""
    echo "[1/5] Cloning whisper.cpp repository..."
    cd "$PROJECT_ROOT"
    git clone https://github.com/ggerganov/whisper.cpp.git
    echo "✓ Repository cloned"
else
    echo ""
    echo "[1/5] whisper.cpp already exists, pulling latest..."
    cd "$WHISPER_DIR"
    git pull
    echo "✓ Repository updated"
fi

# Step 2: Download Distil-Whisper Small model
echo ""
echo "[2/5] Downloading Distil-Whisper Small model..."
cd "$WHISPER_DIR"

if [ ! -f "models/ggml-distil-small.en.bin" ]; then
    bash ./models/download-ggml-model.sh distil-small.en
    echo "✓ Model downloaded"
else
    echo "✓ Model already exists"
fi

# Step 3: Build whisper.cpp Android library
echo ""
echo "[3/5] Building Android library..."
cd "$WHISPER_DIR"

# Check if Android NDK is available
if [ -z "$ANDROID_NDK_HOME" ] && [ -z "$ANDROID_NDK" ]; then
    echo "⚠ Warning: ANDROID_NDK_HOME not set"
    echo "Please install Android NDK and set ANDROID_NDK_HOME environment variable"
    echo "Example: export ANDROID_NDK_HOME=$HOME/Library/Android/sdk/ndk/26.1.10909125"
    exit 1
fi

# Use NDK path
NDK_PATH="${ANDROID_NDK_HOME:-$ANDROID_NDK}"
echo "Using NDK: $NDK_PATH"

# Build for arm64-v8a (Galaxy S25 Ultra)
if [ ! -d "build-android" ]; then
    mkdir build-android
fi

cd build-android
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE="$NDK_PATH/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-31 \
    -DCMAKE_BUILD_TYPE=Release \
    -DWHISPER_BUILD_EXAMPLES=OFF

cmake --build . -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)
echo "✓ Library built"

# Step 4: Copy model to Android assets
echo ""
echo "[4/5] Copying model to Android assets..."
mkdir -p "$MODELS_DIR"
cp "$WHISPER_DIR/models/ggml-distil-small.en.bin" "$MODELS_DIR/"

MODEL_SIZE=$(du -h "$MODELS_DIR/ggml-distil-small.en.bin" | cut -f1)
echo "✓ Model copied: $MODEL_SIZE"

# Step 5: Copy native library to Android project
echo ""
echo "[5/5] Copying native library..."
LIBS_DIR="$PROJECT_ROOT/app/src/main/jniLibs/arm64-v8a"
mkdir -p "$LIBS_DIR"

# Find and copy the whisper library
if [ -f "$WHISPER_DIR/build-android/src/libwhisper.so" ]; then
    cp "$WHISPER_DIR/build-android/src/libwhisper.so" "$LIBS_DIR/"
    echo "✓ libwhisper.so copied"
else
    echo "⚠ Warning: libwhisper.so not found, trying alternative path..."
    find "$WHISPER_DIR/build-android" -name "libwhisper.so" -exec cp {} "$LIBS_DIR/" \;
fi

LIB_SIZE=$(du -h "$LIBS_DIR/libwhisper.so" 2>/dev/null | cut -f1 || echo "N/A")
echo "✓ Native library size: $LIB_SIZE"

echo ""
echo "=========================================="
echo "✅ WHISPER.CPP SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Files created:"
echo "  Model: $MODELS_DIR/ggml-distil-small.en.bin ($MODEL_SIZE)"
echo "  Library: $LIBS_DIR/libwhisper.so ($LIB_SIZE)"
echo ""
echo "Next steps:"
echo "  1. Add JNI wrapper to Android project"
echo "  2. Update AsrSession.kt to use whisper.cpp"
echo "  3. Build and test on device"
echo ""

