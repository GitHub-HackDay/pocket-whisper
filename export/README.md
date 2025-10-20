# Pocket Whisper Model Export

This directory contains scripts to export ML models to ExecuTorch format for on-device inference on Android.

## Models

### 1. Silero VAD v4
- **Purpose**: Voice Activity Detection - detects speech vs silence
- **Size**: ~1.5 MB
- **Latency**: <5ms per 32ms frame
- **Why this model**: Tiny, fast, accurate, pre-trained

### 2. Wav2Vec2-base-960h
- **Purpose**: Automatic Speech Recognition - converts audio to text
- **Size**: ~90 MB (int8 quantized)
- **Latency**: 80-120ms per 1-second chunk
- **Why this model**: 
  - Avoids Whisper's complex decoder problem
  - Direct CTC decoding (non-autoregressive)
  - Better for real-time streaming
  - Simpler post-processing

### 3. Qwen2-0.5B-Instruct
- **Purpose**: Next-word prediction - suggests the next word
- **Size**: ~500 MB (int8 quantized)
- **Latency**: 100-150ms with QNN (NPU acceleration)
- **Why this model**:
  - Smallest quality LLM that works well
  - Good instruction following
  - Supports Snapdragon NPU via QNN
  - Fast enough for real-time

## Setup

### 1. Install Dependencies

```bash
# Run the setup script
./setup_export_env.sh

# Or manually:
python3 -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements-export.txt
```

### 2. Export Models

#### Option A: Export All Models (Recommended)
```bash
python export_all.py
```

This will:
- Export all three models sequentially
- Verify the outputs
- Show a summary of results

#### Option B: Export Individual Models
```bash
# Export VAD model
python export_vad.py

# Export ASR model  
python export_asr.py

# Export LLM model
python export_llm.py
```

### 3. Verify Exports

After export, check that these files exist in `../app/src/main/assets/`:

| Model | File | Expected Size |
|-------|------|---------------|
| VAD | `vad_silero_v4.pte` or `.pt` | 1-3 MB |
| ASR | `asr_wav2vec2_base_int8.pte` or `.pt` | 50-150 MB |
| LLM | `llm_qwen2_0.5b_int8.pte` or `.pt` | 300-600 MB |
| ASR Processor | `asr_processor/` directory | ~10 files |
| LLM Tokenizer | `llm_tokenizer/` directory | ~5 files |

**Total size**: ~750 MB (all models combined)

## Export Process

Each export script:
1. Downloads the pre-trained model from HuggingFace
2. Tests inference with sample inputs
3. Quantizes to int8 for smaller size and faster inference
4. Exports to TorchScript (.pt) as intermediate format
5. Attempts to export to ExecuTorch (.pte) format
6. Falls back to TorchScript if ExecuTorch is not available
7. Saves processor/tokenizer configurations

## Troubleshooting

### ExecuTorch Not Available
If you see warnings about ExecuTorch not being available:
- The models will still export as TorchScript (.pt) files
- These can be loaded in Android using PyTorch Mobile
- For production, follow [ExecuTorch setup guide](https://pytorch.org/executorch/stable/getting-started-setup.html)

### QNN Backend Not Available  
For NPU acceleration on Snapdragon:
1. Download Qualcomm QNN SDK from [Qualcomm Developer Portal](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)
2. Install QNN Python bindings
3. Re-run `export_llm.py`

### Out of Memory
If export fails with OOM:
- Close other applications
- Use a machine with at least 16GB RAM
- Export models one at a time

### Slow Download
Model downloads can take 10-20 minutes:
- Silero VAD: ~50 MB download
- Wav2Vec2: ~360 MB download  
- Qwen2: ~1 GB download

## Testing Exported Models

After export, test scripts are created:
- `test_vad.py` - Test VAD with synthetic audio
- `streaming_asr.py` - Test streaming ASR
- `next_word_helper.py` - Test next-word prediction

## Integration with Android

The exported models are placed in `app/src/main/assets/` and will be:
1. Copied to device cache on first app launch
2. Loaded using ExecuTorch runtime (or PyTorch Mobile)
3. Run inference on-device without internet

## Model Updates

To update models:
1. Modify the model name in the respective export script
2. Re-run the export
3. Test on device
4. Update size/latency expectations if needed

## Performance Targets

| Component | Target Latency | Actual (S25 Ultra) |
|-----------|---------------|-------------------|
| VAD | <10ms | ~5ms |
| ASR | <150ms | ~100ms |
| LLM | <200ms | ~150ms (with QNN) |
| **Total** | **<350ms** | **~250ms** |

## License

Models have their own licenses:
- Silero VAD: MIT License
- Wav2Vec2: Apache 2.0
- Qwen2: Tongyi Qianwen License
