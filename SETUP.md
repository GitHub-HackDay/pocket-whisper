# Pocket Whisper - Detailed Setup Guide

This guide walks you through setting up the development environment for Pocket Whisper.

## Prerequisites

- **Operating System**: macOS, Linux, or Windows 10+
- **Python**: 3.10 or higher
- **Storage**: ~10GB free space (for models and dependencies)
- **RAM**: 16GB recommended (8GB minimum)
- **Android Development**: Android Studio (for Phase 2+)
- **S25 Ultra**: Galaxy S25 Ultra device or emulator

## Step-by-Step Setup

### 1. Clone Repository

```bash
cd ~/Documents/Projects
git clone <repository-url> pocket-whisper
cd pocket-whisper
```

### 2. Create Python Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your command prompt.

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch 2.5.0
- ExecuTorch 0.4.0
- Transformers 4.36.0
- TorchAudio 2.5.0
- TorchAO 0.1.0
- Librosa 0.10.1
- SoundFile 0.12.1
- ONNX 1.15.0

**Installation time**: 5-10 minutes depending on internet speed.

### 5. Validate Environment

Run the validation script to ensure everything is installed correctly:

```bash
python scripts/validate_env.py
```

Expected output:
```
======================================================================
Pocket Whisper - Environment Validation
======================================================================

[1/8] Testing PyTorch...
  ✓ PyTorch 2.5.0 imported successfully
  ✓ Basic tensor operations work

[2/8] Testing TorchAudio...
  ✓ TorchAudio 2.5.0 imported successfully

... (more tests)

======================================================================
✅ ALL TESTS PASSED!
Environment is ready for Pocket Whisper development.
======================================================================
```

If you see errors, see the **Troubleshooting** section below.

### 6. Generate Test Audio

Create test audio files for development:

```bash
python scripts/generate_test_audio.py
```

This creates:
- `test_audio/silence.wav` - 2 seconds of silence
- `test_audio/speech.wav` - 3 seconds of simulated speech
- `test_audio/filler.wav` - 2 seconds with filler words

### 7. Export Models

Now you're ready to export the ML models.

**Note**: Model export can take 30-60 minutes total and requires ~10GB disk space.

#### 7.1 Export VAD Model (~2 minutes)

```bash
python export/export_vad.py
```

Output: `app/src/main/assets/vad_silero.pte` (~1.5MB)

#### 7.2 Export ASR Model (~10-15 minutes)

```bash
python export/export_asr.py
```

Output: `app/src/main/assets/asr_distil_whisper_small_int8.pte` (~244MB)

#### 7.3 Export LLM Model (~20-30 minutes)

```bash
python export/export_llm.py
```

Output: `app/src/main/assets/llm_qwen_0.5b_int8_cpu.pte` (~500MB)

#### 7.4 Export Mel Preprocessor (~1 minute)

```bash
python export/export_mel_preprocessor.py
```

Output: `app/src/main/assets/mel_preprocessor.onnx` (~few MB)

### 8. Validate All Models

After exporting, validate that all models work correctly:

```bash
python scripts/validate_all_models.py
```

Expected output:
```
======================================================================
VALIDATING ALL EXPORTED MODELS
======================================================================

[1/3] TESTING VAD MODEL
  ✓ Model loaded
  ✓ VAD validation passed!

[2/3] TESTING ASR MODEL
  ✓ Processor loaded
  ✓ Model loaded
  ✓ ASR validation passed!

[3/3] TESTING LLM MODEL
  ✓ Tokenizer loaded
  ✓ Model loaded
  ✓ LLM validation passed!

======================================================================
✅ ALL MODELS VALIDATED SUCCESSFULLY!
======================================================================
```

## Troubleshooting

### Common Issues

#### 1. "torch" module not found

**Problem**: PyTorch not installed correctly.

**Solution**:
```bash
pip uninstall torch torchvision torchaudio
pip install torch==2.5.0 torchvision torchaudio
```

#### 2. "executorch" module not found

**Problem**: ExecuTorch not installed.

**Solution**:
```bash
pip install executorch==0.4.0
```

If that fails, try installing from source:
```bash
git clone https://github.com/pytorch/executorch.git
cd executorch
pip install .
```

#### 3. Out of Memory during export

**Problem**: Not enough RAM to export large models.

**Solutions**:
- Close other applications
- Export models one at a time
- Use a machine with more RAM
- For LLM, you can skip QNN export initially

#### 4. "No module named 'transformers'"

**Problem**: Transformers library not installed.

**Solution**:
```bash
pip install transformers==4.36.0
```

#### 5. CUDA/GPU issues

**Note**: GPU is not required for model export (CPU is fine).

If you want to use GPU:
```bash
# For NVIDIA GPUs
pip install torch==2.5.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

#### 6. Model export takes too long

**Normal times**:
- VAD: 2-5 minutes
- ASR: 10-20 minutes
- LLM: 20-45 minutes

If significantly longer, check:
- CPU usage (should be high during export)
- Disk space (need ~10GB free)
- Internet connection (for model downloads)

### Platform-Specific Issues

#### macOS (Apple Silicon M1/M2/M3)

PyTorch should automatically use MPS (Metal Performance Shaders):

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

If MPS is not available, PyTorch will fallback to CPU (which is fine for export).

#### Windows

- Use PowerShell or CMD (not Git Bash)
- Activate venv with `venv\Scripts\activate`
- If you get execution policy errors:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

#### Linux

Make sure you have Python development headers:
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev

# Fedora/RHEL
sudo dnf install python3-devel
```

## Next Steps

After completing setup:

1. **Verify**: Run `python scripts/validate_all_models.py`
2. **Android Setup**: Follow Phase 2 in `IMPLEMENTATION_GUIDE.md`
3. **Development**: Start building the Android app

## Getting Help

If you encounter issues:

1. Check the **Troubleshooting** section above
2. Run `python scripts/validate_env.py` for diagnostics
3. Check `IMPLEMENTATION_GUIDE.md` for detailed explanations
4. Open a GitHub issue with:
   - Error message
   - Output of `python --version`
   - Output of `pip list`
   - Operating system

## System Requirements Summary

### Minimum
- Python 3.10+
- 8GB RAM
- 10GB free disk space
- Dual-core CPU

### Recommended
- Python 3.11+
- 16GB RAM
- 20GB free disk space
- Quad-core CPU
- SSD (for faster model loading)

### For Android Development (Phase 2+)
- Android Studio Arctic Fox+
- Additional 10GB for Android SDK
- Galaxy S25 Ultra device or emulator
