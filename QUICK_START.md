# Pocket Whisper - Quick Start Guide 🎤

## What is Pocket Whisper?

Pocket Whisper is an on-device speech completion app that helps users complete their sentences when they stumble or forget words during speech. It uses ExecuTorch to run AI models locally on your Samsung S25 Ultra without sending data to the cloud.

**Example:**
- **User says:** "Can you please _um_ me the salt?"
- **App fills:** "pass."

## 🚀 Quick Start (5 minutes)

### 1. Prerequisites
- Samsung S25 Ultra with USB debugging enabled
- Python 3.8+ installed
- Android Studio (for Android development)

### 2. Setup
```bash
# Navigate to the project
cd /Users/warrenfeng/Downloads/pocket-whisper

# Run the complete setup (already done!)
python3 scripts/setup_workspace.py

# Create Android project
python3 scripts/create_android_project.py

# Convert models to ExecuTorch format
python3 scripts/model_converter.py --action create_test
```

### 3. Connect Your Device
1. Enable Developer Options on your Samsung S25 Ultra
2. Enable USB Debugging
3. Connect via USB cable
4. Verify connection: `adb devices`

### 4. Deploy to Device
```bash
# Deploy the app to your Samsung S25 Ultra
python3 scripts/device_deploy.py --action deploy
```

### 5. Start Development
```bash
# Start the development server with hot reloading
python3 scripts/dev_server.py
```

## 🎯 How It Works

1. **Audio Input**: User speaks into the microphone
2. **ASR Processing**: Whisper Tiny model transcribes speech to text
3. **Missing Word Detection**: App identifies filler words like "um", "uh", pauses
4. **Word Prediction**: Local language model predicts the most likely missing word
5. **Text Completion**: App displays the completed sentence

## 📱 App Features

- **Real-time Processing**: Instant word suggestions as you speak
- **Privacy-First**: All processing happens on-device
- **Offline Capable**: Works without internet connection
- **Optimized for Samsung S25 Ultra**: Uses device's hardware acceleration

## 🛠️ Development Workflow

### Using Cursor IDE
1. Open the project in Cursor
2. Use `Ctrl+Shift+P` → "Tasks: Run Task" to access:
   - Setup ExecuTorch Environment
   - Create Android Project
   - Convert Models
   - Deploy to Device
   - Test Speech Completion

### File Structure
```
pocket-whisper/
├── android_app/          # Android application
├── models/              # ExecuTorch model files
├── src/                 # Python source code
│   ├── asr/            # Speech recognition
│   └── completion/     # Word prediction
├── scripts/            # Development scripts
└── data/               # Training and test data
```

## 🧪 Testing

```bash
# Test the complete pipeline
python3 scripts/test_speech_completion.py --test pipeline

# Test with sample data
python3 scripts/test_speech_completion.py --test sample_data

# Benchmark performance
python3 scripts/test_speech_completion.py --test benchmark
```

## 🔧 Customization

### Adding New Models
```bash
# Convert your own ASR model
python3 scripts/model_converter.py \
    --action convert \
    --model_path your_model.pt \
    --output_path models/asr/your_model.pte \
    --model_type asr

# Convert your own completion model
python3 scripts/model_converter.py \
    --action convert \
    --model_path your_completion_model.pt \
    --output_path models/completion/your_model.pte \
    --model_type completion
```

### Modifying the App
- **Android UI**: Edit files in `android_app/app/src/main/`
- **ASR Logic**: Modify `src/asr/whisper_asr.py`
- **Completion Logic**: Modify `src/completion/word_predictor.py`

## 📊 Performance Optimization

### For Samsung S25 Ultra
- Uses `arm64-v8a` architecture
- Optimized for XNNPACK backend
- GPU acceleration with Vulkan backend
- Quantized models for better performance

### Model Optimization
- **Quantization**: INT8 quantization for faster inference
- **Pruning**: Remove unnecessary model weights
- **Knowledge Distillation**: Smaller models trained from larger ones

## 🐛 Troubleshooting

### Common Issues

1. **ADB not found**
   ```bash
   # Install Android SDK and add to PATH
   export PATH=$PATH:$ANDROID_HOME/platform-tools
   ```

2. **Device not detected**
   - Enable USB Debugging
   - Check USB connection
   - Run: `adb devices`

3. **ExecuTorch not found**
   - See `EXECUTORCH_SETUP.md` for installation instructions
   - Install from source: `git clone https://github.com/pytorch/executorch.git`

4. **Build failures**
   - Check Android SDK version
   - Ensure Gradle is properly configured
   - Check device compatibility

## 📚 Next Steps

1. **Experiment with Models**: Try different ASR and completion models
2. **Collect Data**: Gather speech data for training custom models
3. **Optimize Performance**: Fine-tune models for your specific use case
4. **Add Features**: Implement additional speech processing capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test on Samsung S25 Ultra
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

---

**Happy coding! 🎉**

For more detailed information, see the main [README.md](README.md) file.

