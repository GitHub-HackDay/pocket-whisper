# Pocket Whisper

An on-device AI assistant that quietly "whispers" the word you're searching for — helping non-native speakers or older adults recall words in real time through a small earpiece, with all processing done locally.

## Overview

Pocket Whisper is a fully on-device Android application (targeting Galaxy S25 Ultra) that:
- **Listens** to your speech in real-time
- **Detects** when you pause or use filler words ("uh", "um")
- **Suggests** the next word using a local LLM
- **Speaks** the suggestion via TTS and **auto-inserts** it into your active text field
- **Learns** from corrections to improve over time

**All processing happens on-device** — no cloud, works in airplane mode, completely private.

## Key Features

- 🎤 **Real-time Speech Recognition** (Distil-Whisper)
- 🧠 **Next-Word Prediction** (Qwen2-0.5B with QNN acceleration)
- 🎯 **Intelligent Trigger System** (adapts to your speaking rate)
- 🔊 **Audio Output** (TTS via speaker or earpiece)
- ⌨️ **Auto-Insert** (via Accessibility Service)
- 📊 **Feedback Loop** (learns from corrections)
- ✈️ **Fully Offline** (airplane mode compatible)

## Performance Targets

- **Latency**: <350ms P95 (pause → suggestion)
- **Accuracy**: <10% correction rate
- **False Positives**: <2% unwanted insertions
- **Battery**: <5% drain per hour

## Project Structure

```
pocket-whisper/
├── export/                  # Model export scripts
│   ├── export_vad.py       # Voice Activity Detection (Silero VAD)
│   ├── export_asr.py       # Speech Recognition (Distil-Whisper)
│   ├── export_llm.py       # Language Model (Qwen2-0.5B)
│   └── export_mel_preprocessor.py  # Mel spectrogram preprocessor
├── scripts/                 # Development tools
│   ├── validate_env.py     # Environment validation
│   ├── generate_test_audio.py  # Test audio generation
│   ├── validate_all_models.py  # Model validation
│   └── benchmark.py        # Latency profiling
├── app/                     # Android application (to be created)
├── test_audio/             # Test audio files
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Quick Start

### Phase 0: Environment Setup

1. **Create Python environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Validate environment:**
   ```bash
   python scripts/validate_env.py
   ```

4. **Generate test audio:**
   ```bash
   python scripts/generate_test_audio.py
   ```

### Phase 1: Export Models

1. **Export VAD model (~1.5MB):**
   ```bash
   python export/export_vad.py
   ```

2. **Export ASR model (~244MB):**
   ```bash
   python export/export_asr.py
   ```

3. **Export LLM model (~500MB):**
   ```bash
   python export/export_llm.py
   ```

4. **Export mel preprocessor:**
   ```bash
   python export/export_mel_preprocessor.py
   ```

5. **Validate all models:**
   ```bash
   python scripts/validate_all_models.py
   ```

### Phase 2+: Android Development

See `IMPLEMENTATION_GUIDE.md` for detailed Android development instructions.

## Requirements

### Python Environment
- Python 3.10+
- PyTorch 2.5.0
- ExecuTorch 0.4.0
- Transformers 4.36.0
- See `requirements.txt` for full list

### Android Development
- Android Studio (latest)
- Galaxy S25 Ultra or emulator
- Android SDK API 31+
- Kotlin support

### Optional (for QNN acceleration)
- Qualcomm QNN SDK (for NPU acceleration)
- ~180ms LLM latency vs ~500-1000ms without

## Documentation

- **`IMPLEMENTATION_GUIDE.md`** - Complete 15-week implementation plan
- **`pocket_whisper_on_device_next_word_assist_s_25_ultra.md`** - Original specification

## Architecture

```
┌─────────────────┐
│  Audio Pipeline │ → VAD → Filler Detect
└────────┬────────┘
         ↓
┌─────────────────┐
│   ASR (Whisper) │ → Streaming transcription
└────────┬────────┘
         ↓
┌─────────────────┐
│  Trigger System │ → Speaking rate + confidence checks
└────────┬────────┘
         ↓
┌─────────────────┐
│   LLM (Qwen)    │ → Next-word prediction
└────────┬────────┘
         ↓
┌─────────────────┐
│ TTS + Auto-Ins  │ → Audio output + text insertion
└─────────────────┘
```

## Development Status

- [x] Phase 0: Environment Setup
- [x] Phase 1: Model Export Scripts
- [ ] Phase 2: Android Foundation
- [ ] Phase 3: Audio Pipeline + VAD
- [ ] Phase 4: ASR Integration
- [ ] Phase 5: LLM Integration
- [ ] Phase 6: Trigger System
- [ ] Phase 7: Output & Feedback
- [ ] Phase 8: Data Layer & UI
- [ ] Phase 9: Integration & E2E Flow
- [ ] Phase 10: Testing & Optimization

## License

See LICENSE file for details.

## Contributing

This is an active development project. See `IMPLEMENTATION_GUIDE.md` for how to contribute.

## Troubleshooting

### Model Export Issues
- **ExecuTorch not found**: Make sure you installed via `pip install executorch`
- **OOM during export**: Close other applications, export models one at a time
- **QNN export fails**: Optional - you can use CPU fallback

### Environment Issues
- Run `python scripts/validate_env.py` to diagnose
- Make sure Python 3.10+ is being used
- Try creating a fresh virtual environment

## Contact

For questions or issues, please open a GitHub issue.
