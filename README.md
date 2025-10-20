# Pocket Whisper

**On-device next-word assistant for Samsung Galaxy S25 Ultra**

Real-time speech assistance that listens, detects pauses, and auto-suggests the next word - all running locally on your phone with <350ms latency.

---

## 🤖 Quick Reference for LLMs

**Current State (Oct 21, 2025):**
- ✅ **VAD Model**: Fully working on device (Silero VAD, 2-5ms, 70-90% accuracy)
- ✅ **Audio Pipeline**: Real-time capture working (16kHz, 32ms chunks)
- ✅ **PyTorch Mobile**: Integrated and validated
- ⏳ **Next**: Export and integrate ASR (Wav2Vec2) + LLM (Qwen2-0.5B)

**Key Files:**
- `PROJECT_STATUS.md` ← Read this for current progress
- `app/src/main/java/com/pocketwhisper/app/ml/` ← ML integration code
- `export/` ← Model export scripts

**Stack:** Kotlin, PyTorch Mobile, Jetpack Compose, Android 15

---

## 📁 Key Documentation

| Document | Purpose |
|----------|---------|
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | **Current progress, what's done, what's next** |
| [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) | **Complete technical implementation guide** |
| [pocket_whisper_on_device_next_word_assist_s_25_ultra.md](pocket_whisper_on_device_next_word_assist_s_25_ultra.md) | Original specification |

## 🚀 Quick Start

### Current Status: VAD Complete, Moving to ASR/LLM
- ✅ Phase 1: Environment setup complete
- ✅ Phase 2: VAD model complete and working on device!
- 🔄 Phase 3: ASR/LLM export and integration in progress
- ⏳ Phase 4-9: Full pipeline and features

### What's Working Now
- ✅ **VAD Model**: Silero VAD v4 running on device with 70-90% accuracy
- ✅ **Real-time Audio**: Microphone capture with 32ms chunks at 16kHz
- ✅ **PyTorch Mobile**: Full integration and validation complete
- ✅ **Android App**: Functional UI with VAD testing capabilities

### Next Steps
1. **Export ASR model** (speech-to-text):
   ```bash
   cd export
   python export_asr_torchscript.py  # Export Wav2Vec2
   ```

2. **Test ASR on device**:
   - Integrate AsrTranscriber.kt
   - Test with real speech input
   - Validate transcription quality

3. **Export and integrate LLM** (next-word prediction):
   ```bash
   python export_llm_torchscript.py  # Export Qwen2-0.5B
   ```

## 🎯 Project Goals

- **Latency**: <350ms end-to-end (currently projecting ~255ms)
- **Privacy**: 100% on-device processing
- **Accuracy**: >90% ASR, <10% correction rate
- **Models**: VAD + ASR + LLM (~592MB total)

## 🛠️ Tech Stack

- **Models**: PyTorch Mobile (TorchScript `.pt` files)
- **Android**: Kotlin + Jetpack Compose
- **Device**: Samsung Galaxy S25 Ultra (Snapdragon 8 Gen 4)

## 📂 Project Structure

```
pocket-whisper/
├── app/                      # Android app
│   └── src/main/assets/     # Model files (.pt)
├── export/                   # Model export scripts
├── README.md                 # This file
├── PROJECT_STATUS.md         # Current progress tracker
└── IMPLEMENTATION_GUIDE.md   # Full technical guide
```

## 🏃 Development Workflow

1. **Check current status**: Read [PROJECT_STATUS.md](PROJECT_STATUS.md)
2. **Implementation details**: Refer to [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
3. **Export models**: Use scripts in `export/`
4. **Build Android app**: Run `./build_and_deploy.sh`
5. **Test on device**: Check logs with `adb logcat -s PocketWhisper`

---

*For detailed progress and next steps, see [PROJECT_STATUS.md](PROJECT_STATUS.md)*