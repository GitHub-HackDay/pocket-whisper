#!/bin/bash
# Phase 1 Verification Script for Pocket Whisper

echo "========================================"
echo "Pocket Whisper - Phase 1 Verification"
echo "========================================"
echo ""

# Check Python virtual environment
echo "1. Checking Python virtual environment..."
if [ -d ".venv" ]; then
    echo "   ✅ .venv directory exists"
else
    echo "   ❌ .venv directory NOT found"
    exit 1
fi

# Activate venv and check packages
echo ""
echo "2. Checking Python packages..."
source .venv/bin/activate
python -c "
import sys
packages = {
    'torch': 'PyTorch',
    'executorch': 'ExecuTorch',
    'transformers': 'Transformers',
    'librosa': 'Librosa',
    'soundfile': 'SoundFile',
    'torchao': 'TorchAO'
}

all_ok = True
for pkg, name in packages.items():
    try:
        __import__(pkg)
        print(f'   ✅ {name} installed')
    except ImportError:
        print(f'   ❌ {name} NOT installed')
        all_ok = False

sys.exit(0 if all_ok else 1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Some Python packages are missing!"
    exit 1
fi

# Check requirements-ml.txt
echo ""
echo "3. Checking requirements-ml.txt..."
if [ -f "requirements-ml.txt" ]; then
    echo "   ✅ requirements-ml.txt exists"
else
    echo "   ❌ requirements-ml.txt NOT found"
    exit 1
fi

# Check Android project structure
echo ""
echo "4. Checking Android project structure..."

files=(
    "settings.gradle.kts"
    "build.gradle.kts"
    "gradle.properties"
    "app/build.gradle.kts"
    "app/src/main/AndroidManifest.xml"
    "app/src/main/java/com/pocketwhisper/app/MainActivity.kt"
    "app/src/main/java/com/pocketwhisper/app/ml/ModelLoader.kt"
    "app/src/main/res/values/strings.xml"
    "app/src/main/res/values/themes.xml"
    "app/src/main/res/xml/accessibility_service_config.xml"
)

all_files_ok=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file NOT found"
        all_files_ok=false
    fi
done

if [ "$all_files_ok" = false ]; then
    echo ""
    echo "❌ Some Android project files are missing!"
    exit 1
fi

# Check assets directory
echo ""
echo "5. Checking assets directory..."
if [ -d "app/src/main/assets" ]; then
    echo "   ✅ assets/ directory exists (ready for Phase 2 models)"
else
    echo "   ❌ assets/ directory NOT found"
    exit 1
fi

# Final summary
echo ""
echo "========================================"
echo "✅ Phase 1 Verification PASSED!"
echo "========================================"
echo ""
echo "All requirements met:"
echo "  ✅ Python environment with ExecuTorch"
echo "  ✅ Android project structure created"
echo "  ✅ Build configuration ready"
echo "  ✅ Core classes (MainActivity, ModelLoader) created"
echo ""
echo "Next Steps:"
echo "  1. Open project in Android Studio"
echo "  2. Sync Gradle (may take 5-10 minutes)"
echo "  3. Connect S25 Ultra via USB"
echo "  4. Build and run the app"
echo "  5. Proceed to Phase 2: Model Export"
echo ""
echo "See PHASE1_COMPLETE.md for detailed instructions."
echo ""

