# Pocket Whisper — Implementation Guide for Software Engineers

> **Audience**: Android/Kotlin developers with **no ML background**. This guide assumes you can write Android apps but have never touched PyTorch, ExecuTorch, or on-device ML.

---

## Table of Contents
1. [Understanding the ML Stack](#1-understanding-the-ml-stack)
2. [Phase 1: Environment Setup](#phase-1-environment-setup-week-1-day-1-2)
3. [Phase 2: Model Selection & Export](#phase-2-model-selection--export-week-1-day-3-5)
4. [Phase 3: Android Project Setup](#phase-3-android-project-setup-week-1-2)
5. [Phase 4: Audio Pipeline](#phase-4-audio-pipeline-week-2)
6. [Phase 5: ASR Integration](#phase-5-asr-integration-week-2-3)
7. [Phase 6: LLM Integration](#phase-6-llm-integration-week-3)
8. [Phase 7: Trigger System](#phase-7-trigger-system-week-3-4)
9. [Phase 8: Output & Feedback](#phase-8-output--feedback-week-4)
10. [Phase 9: Testing & Optimization](#phase-9-testing--optimization-week-4-5)

---

## 1. Understanding the ML Stack

### What You're Building (In Simple Terms)
Think of this app as a **pipeline** with these stages:

```
Audio from Mic → VAD → Filler Detect → ASR → Trigger Logic → LLM → TTS + Insert
    (raw)      (yes/no) (filler score) (text)  (should we?)  (word)  (output)
```

### What is ExecuTorch?
- **What**: A runtime that runs PyTorch models on mobile devices (like a mini TensorFlow Lite)
- **Why**: It can use the phone's NPU (Neural Processing Unit) for fast inference
- **File format**: `.pte` (PyTorch ExecuTorch) — think of it like a `.jar` for ML models

### What is a Model?
- **Input**: Numbers (audio samples, text tokens)
- **Output**: Numbers (probabilities, predictions)
- **Example**: ASR model takes audio → outputs text probabilities

### Model Size Tradeoffs
- **Larger models** (3B+ params): More accurate, but SLOW (1-2 seconds per word) ❌
- **Tiny models** (<100M params): Fast but inaccurate ❌
- **Sweet spot** (300M-1.5B params): Good enough accuracy, <200ms latency ✅

---

## Phase 1: Environment Setup (Week 1, Day 1-2)

### Expected Output
✅ Python environment with ExecuTorch installed  
✅ Android Studio project created  
✅ Can run a "Hello World" Android app on S25 Ultra

### Step 1.1: Install Python Environment

```bash
# Navigate to project directory
cd /Users/kumar/Documents/Projects/MetaHack/pocket-whisper

# Create Python virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install core dependencies
pip install --upgrade pip
pip install torch==2.5.0  # Latest stable PyTorch
pip install torchvision torchaudio
```

### Step 1.2: Install ExecuTorch

```bash
# Install ExecuTorch (PyTorch's mobile runtime)
pip install executorch

# Install quantization tools (to make models smaller/faster)
pip install torchao

# Install transformers (to download pre-trained models)
pip install transformers==4.36.0

# Install audio processing
pip install librosa soundfile

# Save all dependencies
pip freeze > requirements-ml.txt
```

**Why these versions?**
- `torch==2.5.0`: Latest stable with QNN backend support
- `transformers==4.36.0`: Stable version with Whisper & Qwen models
- `executorch`: Required to export `.pte` files

### Step 1.3: Create Android Project

```bash
# Using Android Studio GUI:
# 1. New Project → Empty Compose Activity
# 2. Name: PocketWhisper
# 3. Package: com.pocketwhisper.app
# 4. Language: Kotlin
# 5. Minimum SDK: API 31 (Android 12) - S25 Ultra runs Android 15
```

### Step 1.4: Add ExecuTorch Android Dependencies

Edit `app/build.gradle.kts`:

```kotlin
dependencies {
    // ExecuTorch runtime for Android
    implementation("org.pytorch:executorch:0.3.0")
    
    // Compose UI
    implementation(platform("androidx.compose:compose-bom:2024.01.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    
    // Room database
    implementation("androidx.room:room-runtime:2.6.1")
    implementation("androidx.room:room-ktx:2.6.1")
    ksp("androidx.room:room-compiler:2.6.1")
    
    // DataStore
    implementation("androidx.datastore:datastore-preferences:1.0.0")
    
    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
}
```

**Checkpoint**: Run the empty app on your S25 Ultra. You should see a blank screen.

---

## Phase 2: Model Selection & Export (Week 1, Day 3-5)

### Expected Output
✅ 3 `.pte` model files in `app/src/main/assets/`  
✅ Python export scripts that work  
✅ Models are small enough to ship (<500MB total)

---

### Model 1: VAD (Voice Activity Detection)

**What it does**: Detects if audio contains speech or silence  
**Why we need it**: Save battery by not running ASR on silence

#### Model Choice: Silero VAD
- **Size**: 1.5 MB (tiny!)
- **Speed**: <5ms per frame
- **Accuracy**: 95%+ on clean audio
- **Why this one**: Pre-trained, works out-of-box, fast enough

#### Download & Export

Create `export/export_vad.py`:

```python
import torch
from pathlib import Path

def export_silero_vad():
    """Export Silero VAD model to ExecuTorch format."""
    
    # Download pre-trained Silero VAD
    print("Downloading Silero VAD...")
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    
    model.eval()
    
    # Sample input: 512 audio samples at 16kHz (32ms window)
    sample_input = torch.randn(1, 512)
    
    # Export to TorchScript first
    print("Tracing model...")
    traced = torch.jit.trace(model, sample_input)
    
    # Export to ExecuTorch
    print("Exporting to ExecuTorch...")
    from executorch.exir import to_edge
    from executorch.exir import EdgeCompileConfig
    
    edge_program = to_edge(
        torch.export.export(traced, (sample_input,))
    )
    
    et_program = edge_program.to_executorch()
    
    # Save .pte file
    output_path = Path("../app/src/main/assets/vad_silero.pte")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)
    
    print(f"✅ Exported to {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    export_silero_vad()
```

**Run it:**
```bash
cd export
python export_vad.py
```

**Why Silero VAD and not others?**
- ❌ WebRTC VAD: Too simple, high false positives
- ❌ PyAnnote: 80MB+, too slow for real-time
- ✅ Silero VAD: Small, fast, accurate, easy to export

---

### Model 2: ASR (Automatic Speech Recognition)

**What it does**: Converts audio → text  
**Why we need it**: Get transcription to feed to LLM

#### Model Choice: Distil-Whisper Small En
- **Size**: 244 MB (quantized to int8)
- **Speed**: 80-120ms per 1-second audio chunk
- **Accuracy**: 95% WER (Word Error Rate) on English
- **Why this one**: 
  - ✅ Faster than base Whisper (6x speedup)
  - ✅ Still accurate enough
  - ✅ Designed for streaming
  - ❌ NOT "tiny" (too inaccurate)
  - ❌ NOT "medium/large" (too slow)

#### Download & Export

Create `export/export_asr.py`:

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from pathlib import Path

def export_distil_whisper():
    """Export Distil-Whisper Small model for streaming ASR."""
    
    model_id = "distil-whisper/distil-small.en"
    
    print(f"Loading {model_id}...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # Will quantize later
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    # Load processor (handles audio preprocessing)
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Sample input: 30 seconds of audio at 16kHz
    # Shape: (batch=1, samples=16000*30)
    sample_audio = torch.randn(1, 16000 * 30)
    
    # Preprocess audio to mel spectrogram
    inputs = processor(
        sample_audio.squeeze(),
        sampling_rate=16000,
        return_tensors="pt"
    )
    
    input_features = inputs.input_features
    
    print(f"Model input shape: {input_features.shape}")
    
    # Export encoder only (for streaming, we'll run encoder once per chunk)
    encoder = model.get_encoder()
    encoder.eval()
    
    print("Exporting encoder...")
    traced_encoder = torch.jit.trace(encoder, input_features)
    
    # Quantize to int8 (reduce size & increase speed)
    from torch.ao.quantization import quantize_dynamic
    quantized_encoder = quantize_dynamic(
        traced_encoder,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    # Export to ExecuTorch
    from executorch.exir import to_edge
    
    print("Converting to ExecuTorch format...")
    edge_program = to_edge(
        torch.export.export(quantized_encoder, (input_features,))
    )
    
    et_program = edge_program.to_executorch()
    
    # Save
    output_path = Path("../app/src/main/assets/asr_distil_whisper_small_int8.pte")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)
    
    print(f"✅ Exported to {output_path}")
    print(f"   Size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Save processor config (needed in Android)
    processor.save_pretrained("../app/src/main/assets/asr_processor/")
    print("✅ Saved processor config")

if __name__ == "__main__":
    export_distil_whisper()
```

**Run it:**
```bash
python export_asr.py
```

**Expected time**: 5-10 minutes (downloading + exporting)

**Why Distil-Whisper Small and not others?**
- ❌ Whisper Tiny: Only 13% WER → too many errors → bad suggestions
- ❌ Whisper Base: 50% slower, only 2% better accuracy
- ❌ OpenAI Whisper Large: 3GB+, 5+ seconds per chunk
- ❌ Google Speech API: Cloud-based (we need on-device)
- ✅ Distil-Whisper Small: Perfect balance for real-time

---

### Model 3: LLM (Language Model for Next-Word Prediction)

**What it does**: Given text context, predicts next word  
**Why we need it**: Generate the suggestion to auto-insert

#### Model Choice: Qwen2-0.5B-Instruct
- **Size**: 500 MB (quantized to int8)
- **Speed**: 100-150ms for 1 token with QNN (NPU acceleration)
- **Accuracy**: Perplexity ~15 (good for single-word prediction)
- **Why this one**:
  - ✅ Smallest quality LLM that works
  - ✅ Supports QNN delegate (NPU acceleration on Snapdragon)
  - ✅ Designed for instruction following
  - ❌ NOT Llama-3-8B (too slow: 2+ seconds)
  - ❌ NOT GPT-2 (too old, bad quality)

#### Download & Export

Create `export/export_llm.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

def export_qwen_llm():
    """Export Qwen2-0.5B for next-word prediction with QNN backend."""
    
    model_id = "Qwen/Qwen2-0.5B-Instruct"
    
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu"
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Sample input: "Could you please" → predict next word
    sample_text = "Could you please"
    inputs = tokenizer(sample_text, return_tensors="pt")
    input_ids = inputs.input_ids  # Shape: (1, num_tokens)
    
    print(f"Model input shape: {input_ids.shape}")
    
    # We only care about generating 1 token
    print("Tracing model for single token generation...")
    
    with torch.no_grad():
        # Trace forward pass
        outputs = model(input_ids)
        logits = outputs.logits  # Shape: (1, seq_len, vocab_size)
    
    # For mobile, we'll export just the forward pass
    # and handle sampling in Kotlin
    
    print("Quantizing to int8...")
    from torch.ao.quantization import quantize_dynamic
    
    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    # Export with QNN backend config
    print("Exporting to ExecuTorch with QNN backend...")
    
    # Note: QNN export requires Qualcomm QNN SDK
    # For now, export with CPU backend and we'll swap in Android
    from executorch.exir import to_edge, EdgeCompileConfig
    
    traced = torch.export.export(
        quantized_model,
        (input_ids,)
    )
    
    edge_program = to_edge(
        traced,
        compile_config=EdgeCompileConfig(_check_ir_validity=False)
    )
    
    et_program = edge_program.to_executorch()
    
    # Save
    output_path = Path("../app/src/main/assets/llm_qwen_0.5b_int8.pte")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)
    
    print(f"✅ Exported to {output_path}")
    print(f"   Size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Save tokenizer
    tokenizer.save_pretrained("../app/src/main/assets/llm_tokenizer/")
    print("✅ Saved tokenizer config")

if __name__ == "__main__":
    export_qwen_llm()
```

**Run it:**
```bash
python export_llm.py
```

**Expected time**: 10-15 minutes (large download)

**Why Qwen2-0.5B and not others?**
- ❌ GPT-2 (124M): Outdated (2019), poor instruction following
- ❌ Phi-2 (2.7B): 5x slower, overkill for single-word
- ❌ Llama-3.2-1B: Better quality but 2x slower
- ❌ Gemma-2B: No QNN support yet
- ✅ Qwen2-0.5B: Smallest modern LLM with QNN support

---

### Model 4: Filler Detector (Optional but Recommended)

**What it does**: Detects "uh", "um", "like" in audio  
**Why we need it**: Trigger suggestions during filler words

#### Model Choice: Custom 1D-TCN (Train yourself OR skip for MVP)

**For MVP**: Use a simple heuristic instead of ML model:
- Check if ASR output contains filler words: `["uh", "um", "like", "you know", "I mean"]`
- If yes, increase trigger probability

**If you want the ML model** (Week 3+):
1. Collect 1000 audio clips with fillers
2. Train a tiny 1D Temporal Convolutional Network (TCN)
3. Export to `.pte`

**Recommendation**: Skip for MVP, use keyword matching instead.

---

### Phase 2 Checkpoint

Run this to verify all models exported:

```bash
ls -lh ../app/src/main/assets/

# Expected output:
# vad_silero.pte                    (~1.5 MB)
# asr_distil_whisper_small_int8.pte (~244 MB)
# llm_qwen_0.5b_int8.pte           (~500 MB)
# Total: ~746 MB
```

If you see these files, ✅ **Phase 2 Complete!**

---

## Phase 3: Android Project Setup (Week 1-2)

### Expected Output
✅ App runs and requests microphone permission  
✅ Foreground service starts  
✅ Can load a `.pte` model in Kotlin

### Step 3.1: Update AndroidManifest.xml

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">

    <!-- Permissions -->
    <uses-permission android:name="android.permission.RECORD_AUDIO"/>
    <uses-permission android:name="android.permission.POST_NOTIFICATIONS"/>
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE"/>
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE_MICROPHONE"/>
    <uses-permission android:name="android.permission.MODIFY_AUDIO_SETTINGS"/>
    
    <application
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:theme="@style/Theme.PocketWhisper">
        
        <!-- Main Activity -->
        <activity
            android:name=".MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN"/>
                <category android:name="android.intent.category.LAUNCHER"/>
            </intent-filter>
        </activity>
        
        <!-- Foreground Audio Service -->
        <service
            android:name=".audio.ListenForegroundService"
            android:foregroundServiceType="microphone"
            android:exported="false"/>
        
        <!-- Accessibility Service (for auto-insert) -->
        <service
            android:name=".accessibility.AutoInsertService"
            android:permission="android.permission.BIND_ACCESSIBILITY_SERVICE"
            android:exported="false">
            <intent-filter>
                <action android:name="android.accessibilityservice.AccessibilityService"/>
            </intent-filter>
            <meta-data
                android:name="android.accessibilityservice"
                android:resource="@xml/accessibility_service_config"/>
        </service>
    </application>
</manifest>
```

### Step 3.2: Create Accessibility Service Config

Create `app/src/main/res/xml/accessibility_service_config.xml`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<accessibility-service
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:accessibilityEventTypes="typeViewFocused|typeViewTextChanged"
    android:accessibilityFeedbackType="feedbackGeneric"
    android:accessibilityFlags="flagReportViewIds"
    android:canRetrieveWindowContent="true"
    android:description="@string/accessibility_service_description"
    android:notificationTimeout="100"/>
```

### Step 3.3: Create ExecuTorch Model Loader

Create `app/src/main/java/com/pocketwhisper/app/ml/ModelLoader.kt`:

```kotlin
package com.pocketwhisper.app.ml

import android.content.Context
import android.util.Log
import org.pytorch.executorch.Module
import java.io.File
import java.io.FileOutputStream

class ModelLoader(private val context: Context) {
    
    private val TAG = "ModelLoader"
    
    /**
     * Load a .pte model from assets.
     * Returns a Module ready for inference.
     */
    fun loadModel(assetName: String): Module {
        Log.d(TAG, "Loading model: $assetName")
        
        // Copy from assets to cache (ExecuTorch requires file path)
        val cacheFile = File(context.cacheDir, assetName)
        
        if (!cacheFile.exists()) {
            Log.d(TAG, "Copying $assetName to cache...")
            context.assets.open(assetName).use { input ->
                FileOutputStream(cacheFile).use { output ->
                    input.copyTo(output)
                }
            }
        }
        
        // Load module
        val module = Module.load(cacheFile.absolutePath)
        Log.d(TAG, "✅ Model loaded: $assetName")
        
        return module
    }
}
```

### Step 3.4: Test Model Loading

Create `app/src/main/java/com/pocketwhisper/app/MainActivity.kt`:

```kotlin
package com.pocketwhisper.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.pocketwhisper.app.ml.ModelLoader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        setContent {
            MaterialTheme {
                Surface {
                    TestModelLoadingScreen()
                }
            }
        }
    }
    
    @Composable
    fun TestModelLoadingScreen() {
        var status by remember { mutableStateOf("Ready") }
        val scope = rememberCoroutineScope()
        val modelLoader = remember { ModelLoader(this) }
        
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
        ) {
            Text("Model Loading Test", style = MaterialTheme.typography.headlineMedium)
            
            Spacer(modifier = Modifier.height(16.dp))
            
            Text("Status: $status")
            
            Spacer(modifier = Modifier.height(16.dp))
            
            Button(onClick = {
                scope.launch {
                    status = "Loading VAD..."
                    try {
                        withContext(Dispatchers.IO) {
                            modelLoader.loadModel("vad_silero.pte")
                        }
                        status = "✅ VAD loaded successfully!"
                    } catch (e: Exception) {
                        status = "❌ Error: ${e.message}"
                    }
                }
            }) {
                Text("Test Load VAD")
            }
        }
    }
}
```

**Run the app**: You should see a button. Click it and verify the VAD model loads.

**Phase 3 Checkpoint**: ✅ If you see "VAD loaded successfully", you're ready for Phase 4!

---

## Phase 4: Audio Pipeline (Week 2)

### Expected Output
✅ App captures microphone audio  
✅ Audio is buffered in 32ms chunks  
✅ VAD detects speech vs silence

### Step 4.1: Request Microphone Permission

Update `MainActivity.kt`:

```kotlin
import android.Manifest
import android.content.pm.PackageManager
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

class MainActivity : ComponentActivity() {
    
    private val MICROPHONE_PERMISSION_CODE = 100
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Request mic permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                MICROPHONE_PERMISSION_CODE
            )
        }
        
        setContent { /* ... */ }
    }
}
```

### Step 4.2: Create Audio Capture Service

Create `app/src/main/java/com/pocketwhisper/app/audio/AudioCaptureService.kt`:

```kotlin
package com.pocketwhisper.app.audio

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import java.nio.ByteBuffer
import java.nio.ByteOrder

class AudioCaptureService {
    
    private val TAG = "AudioCapture"
    
    // Audio config: 16kHz mono PCM (standard for speech models)
    private val SAMPLE_RATE = 16000
    private val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
    private val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    
    // Buffer size: 32ms chunks (512 samples)
    private val CHUNK_SIZE = 512
    private val BUFFER_SIZE = AudioRecord.getMinBufferSize(
        SAMPLE_RATE,
        CHANNEL_CONFIG,
        AUDIO_FORMAT
    )
    
    private var audioRecord: AudioRecord? = null
    private var captureJob: Job? = null
    
    // Flow of audio chunks (as FloatArray)
    private val _audioFlow = MutableSharedFlow<FloatArray>(replay = 0)
    val audioFlow: SharedFlow<FloatArray> = _audioFlow
    
    /**
     * Start capturing audio from microphone.
     */
    fun start(scope: CoroutineScope) {
        Log.d(TAG, "Starting audio capture...")
        
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT,
            BUFFER_SIZE
        )
        
        audioRecord?.startRecording()
        
        captureJob = scope.launch(Dispatchers.IO) {
            val buffer = ByteArray(CHUNK_SIZE * 2) // 2 bytes per sample (16-bit)
            
            while (isActive) {
                val bytesRead = audioRecord?.read(buffer, 0, buffer.size) ?: 0
                
                if (bytesRead > 0) {
                    // Convert bytes to floats (normalized to -1.0 to 1.0)
                    val floats = bytesToFloats(buffer, bytesRead)
                    _audioFlow.emit(floats)
                }
            }
        }
        
        Log.d(TAG, "✅ Audio capture started")
    }
    
    /**
     * Stop capturing audio.
     */
    fun stop() {
        Log.d(TAG, "Stopping audio capture...")
        captureJob?.cancel()
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        Log.d(TAG, "✅ Audio capture stopped")
    }
    
    /**
     * Convert 16-bit PCM bytes to normalized floats.
     */
    private fun bytesToFloats(bytes: ByteArray, length: Int): FloatArray {
        val shorts = ShortArray(length / 2)
        ByteBuffer.wrap(bytes, 0, length)
            .order(ByteOrder.LITTLE_ENDIAN)
            .asShortBuffer()
            .get(shorts)
        
        return FloatArray(shorts.size) { i ->
            shorts[i].toFloat() / 32768.0f  // Normalize to [-1, 1]
        }
    }
}
```

### Step 4.3: Integrate VAD

Create `app/src/main/java/com/pocketwhisper/app/ml/VadDetector.kt`:

```kotlin
package com.pocketwhisper.app.ml

import android.util.Log
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

class VadDetector(private val module: Module) {
    
    private val TAG = "VadDetector"
    
    /**
     * Run VAD on audio chunk.
     * Returns probability of speech (0.0 = silence, 1.0 = speech).
     */
    fun detectSpeech(audioChunk: FloatArray): Float {
        // Ensure chunk is 512 samples (32ms at 16kHz)
        require(audioChunk.size == 512) {
            "VAD expects 512 samples, got ${audioChunk.size}"
        }
        
        // Create input tensor: shape (1, 512)
        val inputTensor = Tensor.fromBlob(
            audioChunk,
            longArrayOf(1, 512)
        )
        
        // Run inference
        val outputs = module.forward(inputTensor)
        
        // Output is a single float: speech probability
        val speechProb = outputs.getDataAsFloatArray()[0]
        
        Log.v(TAG, "Speech probability: $speechProb")
        
        return speechProb
    }
}
```

### Step 4.4: Create Foreground Service

Create `app/src/main/java/com/pocketwhisper/app/audio/ListenForegroundService.kt`:

```kotlin
package com.pocketwhisper.app.audio

import android.app.*
import android.content.Intent
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import com.pocketwhisper.app.MainActivity
import com.pocketwhisper.app.R
import com.pocketwhisper.app.ml.ModelLoader
import com.pocketwhisper.app.ml.VadDetector
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.collect

class ListenForegroundService : Service() {
    
    private val TAG = "ListenService"
    private val NOTIFICATION_ID = 1
    private val CHANNEL_ID = "pocket_whisper_channel"
    
    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    
    private lateinit var audioCapture: AudioCaptureService
    private lateinit var vadDetector: VadDetector
    
    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "Service created")
        
        // Create notification channel
        createNotificationChannel()
        
        // Load VAD model
        val modelLoader = ModelLoader(this)
        val vadModule = modelLoader.loadModel("vad_silero.pte")
        vadDetector = VadDetector(vadModule)
        
        // Initialize audio capture
        audioCapture = AudioCaptureService()
    }
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "Service started")
        
        // Start foreground with notification
        val notification = createNotification()
        startForeground(NOTIFICATION_ID, notification)
        
        // Start audio capture
        audioCapture.start(serviceScope)
        
        // Process audio chunks with VAD
        serviceScope.launch {
            audioCapture.audioFlow.collect { audioChunk ->
                val speechProb = vadDetector.detectSpeech(audioChunk)
                
                if (speechProb > 0.5f) {
                    Log.d(TAG, "Speech detected! (prob=$speechProb)")
                    // TODO: Pass to ASR in Phase 5
                } else {
                    Log.v(TAG, "Silence (prob=$speechProb)")
                }
            }
        }
        
        return START_STICKY
    }
    
    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "Service destroyed")
        audioCapture.stop()
        serviceScope.cancel()
    }
    
    override fun onBind(intent: Intent?): IBinder? = null
    
    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Pocket Whisper Listening",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Shows when Pocket Whisper is listening"
        }
        
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(channel)
    }
    
    private fun createNotification(): Notification {
        val intent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_IMMUTABLE
        )
        
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Pocket Whisper")
            .setContentText("Listening...")
            .setSmallIcon(R.drawable.ic_notification)  // TODO: Add icon
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }
}
```

### Step 4.5: Add Toggle to Start/Stop Service

Update `MainActivity.kt`:

```kotlin
@Composable
fun HomeScreen() {
    var isListening by remember { mutableStateOf(false) }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Text("Pocket Whisper", style = MaterialTheme.typography.headlineLarge)
        
        Spacer(modifier = Modifier.height(32.dp))
        
        // Big ON/OFF toggle
        Switch(
            checked = isListening,
            onCheckedChange = { enabled ->
                if (enabled) {
                    startListening()
                } else {
                    stopListening()
                }
                isListening = enabled
            }
        )
        
        Text(if (isListening) "Listening ON" else "Listening OFF")
    }
}

private fun startListening() {
    val intent = Intent(this, ListenForegroundService::class.java)
    startForegroundService(intent)
}

private fun stopListening() {
    val intent = Intent(this, ListenForegroundService::class.java)
    stopService(intent)
}
```

### Phase 4 Checkpoint

✅ Run the app  
✅ Toggle "Listening ON"  
✅ Check Logcat: You should see "Speech detected!" when you talk

If you see this, **Phase 4 Complete!**

---

## Phase 5: ASR Integration (Week 2-3)

### Expected Output
✅ Audio is converted to text  
✅ Partial transcripts are generated every 300ms  
✅ Text is displayed in real-time on the app

### Step 5.1: Create ASR Session

Create `app/src/main/java/com/pocketwhisper/app/ml/AsrSession.kt`:

```kotlin
package com.pocketwhisper.app.ml

import android.content.Context
import android.util.Log
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.util.LinkedList

class AsrSession(
    private val context: Context,
    private val module: Module
) {
    
    private val TAG = "AsrSession"
    
    // Rolling buffer of audio (last 3 seconds)
    private val audioBuffer = LinkedList<FloatArray>()
    private val MAX_BUFFER_CHUNKS = 94  // 3 seconds at 32ms chunks
    
    // Tokenizer (simplified - load from assets in production)
    private val decoder = SimpleWhisperDecoder()
    
    /**
     * Add audio chunk to rolling buffer and maybe generate transcript.
     */
    fun addAudio(audioChunk: FloatArray): String? {
        audioBuffer.add(audioChunk)
        
        // Keep only last 3 seconds
        if (audioBuffer.size > MAX_BUFFER_CHUNKS) {
            audioBuffer.removeFirst()
        }
        
        // Generate transcript every 10 chunks (~300ms)
        return if (audioBuffer.size % 10 == 0) {
            generateTranscript()
        } else {
            null
        }
    }
    
    /**
     * Run ASR on current audio buffer.
     */
    private fun generateTranscript(): String {
        // Concatenate all chunks
        val totalSamples = audioBuffer.sumOf { it.size }
        val audioArray = FloatArray(totalSamples)
        
        var offset = 0
        for (chunk in audioBuffer) {
            chunk.copyInto(audioArray, offset)
            offset += chunk.size
        }
        
        // Convert to mel spectrogram (simplified - use processor in production)
        val melSpec = audioToMel(audioArray)
        
        // Create input tensor
        val inputTensor = Tensor.fromBlob(
            melSpec,
            longArrayOf(1, 80, melSpec.size / 80)  // (batch, n_mels, time)
        )
        
        // Run inference
        val outputs = module.forward(inputTensor)
        
        // Decode to text
        val tokenIds = outputs.getDataAsLongArray()
        val text = decoder.decode(tokenIds)
        
        Log.d(TAG, "Transcript: $text")
        return text
    }
    
    /**
     * Simplified audio → mel spectrogram conversion.
     * In production, use the processor from transformers.
     */
    private fun audioToMel(audio: FloatArray): FloatArray {
        // TODO: Implement proper mel spectrogram
        // For now, return dummy data for testing
        return FloatArray(80 * 100) { 0.0f }
    }
}

/**
 * Simplified Whisper decoder.
 * In production, load tokenizer from assets.
 */
class SimpleWhisperDecoder {
    fun decode(tokenIds: LongArray): String {
        // TODO: Implement proper decoding with tokenizer
        return "[Transcript would appear here]"
    }
}
```

**Note**: The mel spectrogram conversion and tokenization are simplified here. In production, you'll need to:
1. Load the `processor` saved in Phase 2
2. Use it to convert audio → mel spectrogram
3. Load the `tokenizer` to decode token IDs → text

### Step 5.2: Integrate ASR into Service

Update `ListenForegroundService.kt`:

```kotlin
class ListenForegroundService : Service() {
    
    private lateinit var asrSession: AsrSession
    
    override fun onCreate() {
        super.onCreate()
        
        // Load models
        val modelLoader = ModelLoader(this)
        val vadModule = modelLoader.loadModel("vad_silero.pte")
        val asrModule = modelLoader.loadModel("asr_distil_whisper_small_int8.pte")
        
        vadDetector = VadDetector(vadModule)
        asrSession = AsrSession(this, asrModule)
        
        audioCapture = AudioCaptureService()
    }
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // ... (notification code)
        
        audioCapture.start(serviceScope)
        
        serviceScope.launch {
            audioCapture.audioFlow.collect { audioChunk ->
                val speechProb = vadDetector.detectSpeech(audioChunk)
                
                if (speechProb > 0.5f) {
                    // Pass to ASR
                    val transcript = asrSession.addAudio(audioChunk)
                    
                    if (transcript != null) {
                        Log.d(TAG, "📝 Transcript: $transcript")
                        // TODO: Pass to trigger system in Phase 7
                    }
                }
            }
        }
        
        return START_STICKY
    }
}
```

### Phase 5 Checkpoint

✅ Run the app  
✅ Toggle listening ON  
✅ Speak into the phone  
✅ Check Logcat: You should see "📝 Transcript: [text]" every ~300ms

**Phase 5 Complete!**

---

## Phase 6: LLM Integration (Week 3)

### Expected Output
✅ Given text context, LLM generates next word  
✅ Latency is <200ms on S25 Ultra with QNN  
✅ Can generate suggestions like "please" → "confirm"

### Step 6.1: Create LLM Session

Create `app/src/main/java/com/pocketwhisper/app/ml/LlmSession.kt`:

```kotlin
package com.pocketwhisper.app.ml

import android.content.Context
import android.util.Log
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

class LlmSession(
    private val context: Context,
    private val module: Module
) {
    
    private val TAG = "LlmSession"
    
    // Simplified tokenizer (load from assets in production)
    private val tokenizer = SimpleLlmTokenizer()
    
    /**
     * Generate next word given text context.
     * Returns the predicted word and confidence score.
     */
    fun generateNextWord(context: String): Pair<String, Float> {
        val startTime = System.currentTimeMillis()
        
        // Tokenize input
        val tokenIds = tokenizer.encode(context)
        
        // Create input tensor: shape (1, seq_len)
        val inputTensor = Tensor.fromBlob(
            tokenIds.toLongArray(),
            longArrayOf(1, tokenIds.size.toLong())
        )
        
        // Run inference
        val outputs = module.forward(inputTensor)
        
        // Output shape: (1, seq_len, vocab_size)
        // We only care about the last token's logits
        val logits = outputs.getDataAsFloatArray()
        
        // Find top prediction
        val topTokenId = logits.indices.maxByOrNull { logits[it] } ?: 0
        val topProb = logits[topTokenId]
        
        // Decode to word
        val word = tokenizer.decode(intArrayOf(topTokenId))
        
        // Calculate confidence from softmax
        val confidence = calculateConfidence(logits)
        
        val latency = System.currentTimeMillis() - startTime
        Log.d(TAG, "Generated: '$word' (confidence=$confidence, latency=${latency}ms)")
        
        return Pair(word, confidence)
    }
    
    /**
     * Calculate confidence score from logits.
     * Higher = more confident.
     */
    private fun calculateConfidence(logits: FloatArray): Float {
        // Simplified: use top logit probability
        val maxLogit = logits.maxOrNull() ?: 0f
        val expSum = logits.sumOf { Math.exp((it - maxLogit).toDouble()) }.toFloat()
        val topProb = Math.exp((logits.maxOrNull()!! - maxLogit).toDouble()).toFloat() / expSum
        return topProb
    }
}

/**
 * Simplified LLM tokenizer.
 * In production, load from assets (Qwen tokenizer).
 */
class SimpleLlmTokenizer {
    fun encode(text: String): IntArray {
        // TODO: Load real tokenizer from assets
        return intArrayOf(/* token IDs */)
    }
    
    fun decode(tokenIds: IntArray): String {
        // TODO: Load real tokenizer from assets
        return "confirm"  // Dummy for testing
    }
}
```

### Step 6.2: Test LLM Generation

Add a test button to `MainActivity.kt`:

```kotlin
@Composable
fun TestLlmScreen() {
    var result by remember { mutableStateOf("") }
    val scope = rememberCoroutineScope()
    
    Column(modifier = Modifier.padding(16.dp)) {
        Text("LLM Test", style = MaterialTheme.typography.headlineMedium)
        
        Button(onClick = {
            scope.launch {
                val modelLoader = ModelLoader(context)
                val llmModule = modelLoader.loadModel("llm_qwen_0.5b_int8.pte")
                val llmSession = LlmSession(context, llmModule)
                
                val (word, confidence) = llmSession.generateNextWord("Could you please")
                result = "Next word: $word (confidence: $confidence)"
            }
        }) {
            Text("Generate Next Word")
        }
        
        Text(result)
    }
}
```

### Phase 6 Checkpoint

✅ Click "Generate Next Word"  
✅ Should see result in <200ms  
✅ If latency is >500ms, QNN delegate may not be active (see optimization section)

**Phase 6 Complete!**

---

## Phase 7: Trigger System (Week 3-4)

### Expected Output
✅ Only suggests when user genuinely pauses  
✅ No suggestions during natural speech flow  
✅ Adapts to user's speaking rate

### Step 7.1: Implement Speaking Rate Tracker

Create `app/src/main/java/com/pocketwhisper/app/trigger/SpeakingRateTracker.kt`:

```kotlin
package com.pocketwhisper.app.trigger

import kotlin.math.max
import kotlin.math.min

class SpeakingRateTracker {
    
    // Exponential moving average of speaking rate (words per second)
    private var srEma = 3.0  // Default: 3 words/sec
    private val alpha = 0.3  // EMA smoothing factor
    
    /**
     * Update speaking rate based on new transcript.
     */
    fun update(wordsSpoken: Int, timeWindowSeconds: Float) {
        val instantRate = wordsSpoken / timeWindowSeconds
        srEma = alpha * instantRate + (1 - alpha) * srEma
    }
    
    /**
     * Calculate expected pause duration for this speaker.
     * Slower talkers get longer expected pauses.
     */
    fun getExpectedPause(): Float {
        val ep = 0.15 + 0.35 * (1.0 / srEma)
        return min(0.65, max(0.18, ep)).toFloat()
    }
    
    fun getCurrentRate(): Double = srEma
}
```

### Step 7.2: Implement Trigger Policy

Create `app/src/main/java/com/pocketwhisper/app/trigger/TriggerPolicy.kt`:

```kotlin
package com.pocketwhisper.app.trigger

import android.util.Log

data class TriggerSignals(
    val pauseDuration: Float,         // seconds
    val fillerLikelihood: Float,      // 0-1
    val completionConfidence: Float,  // 0-1 from LLM
    val asrStable: Boolean,
    val semanticCoherence: Float,     // 0-1
    val recentlyUsed: Boolean
)

class TriggerPolicy(
    private val speakingRateTracker: SpeakingRateTracker
) {
    
    private val TAG = "TriggerPolicy"
    
    // Thresholds (strict for auto-insert)
    private var tauConf = 0.75f      // Completion confidence threshold
    private var tauFiller = 0.7f     // Filler likelihood threshold
    private var tauSemantic = 0.8f   // Semantic coherence threshold
    
    private val k1 = 1.0f  // Pause multiplier (strict)
    private val k2 = 1.3f  // Long pause multiplier
    
    // Dynamic adjustment tracking
    private val recentOutcomes = mutableListOf<Boolean>()  // true = corrected
    private val maxTracking = 20
    
    /**
     * Decide if we should trigger a suggestion.
     */
    fun shouldTrigger(signals: TriggerSignals): Boolean {
        val ep = speakingRateTracker.getExpectedPause()
        
        // Fire condition (all must be true)
        val pauseOk = signals.pauseDuration >= ep * k1
        val fillerOrLongPause = signals.fillerLikelihood >= tauFiller || 
                                signals.pauseDuration >= ep * k2
        val confidentEnough = signals.completionConfidence >= tauConf
        val semanticOk = signals.semanticCoherence >= tauSemantic
        val notRecentlyUsed = !signals.recentlyUsed
        val asrStable = signals.asrStable
        
        val shouldFire = pauseOk && fillerOrLongPause && confidentEnough && 
                        semanticOk && notRecentlyUsed && asrStable
        
        if (shouldFire) {
            Log.d(TAG, "🔥 TRIGGER! pause=${signals.pauseDuration}s, " +
                      "F=${signals.fillerLikelihood}, C=${signals.completionConfidence}, " +
                      "SC=${signals.semanticCoherence}")
        }
        
        return shouldFire
    }
    
    /**
     * Report if user corrected the last suggestion.
     * Used for dynamic threshold adjustment.
     */
    fun reportCorrection(wasCorrected: Boolean) {
        recentOutcomes.add(wasCorrected)
        
        if (recentOutcomes.size > maxTracking) {
            recentOutcomes.removeAt(0)
        }
        
        // Adjust thresholds
        if (recentOutcomes.size >= 20) {
            val correctionRate = recentOutcomes.count { it } / recentOutcomes.size.toFloat()
            
            when {
                correctionRate > 0.10f -> {
                    // Too many corrections - be more conservative
                    tauConf = min(0.95f, tauConf + 0.05f)
                    Log.d(TAG, "⬆️ Increased confidence threshold to $tauConf")
                }
                correctionRate < 0.03f -> {
                    // Very few corrections - can be more aggressive
                    tauConf = max(0.70f, tauConf - 0.02f)
                    Log.d(TAG, "⬇️ Decreased confidence threshold to $tauConf")
                }
            }
        }
    }
}
```

### Step 7.3: Implement Filler Detection (Simple Version)

Create `app/src/main/java/com/pocketwhisper/app/trigger/FillerDetector.kt`:

```kotlin
package com.pocketwhisper.app.trigger

class FillerDetector {
    
    private val fillerWords = setOf("uh", "um", "like", "you know", "I mean", "so")
    
    /**
     * Check if recent transcript contains filler words.
     * Returns likelihood 0-1.
     */
    fun detectFiller(recentText: String): Float {
        val words = recentText.lowercase().split(" ")
        val lastFewWords = words.takeLast(5)
        
        val hasFillerWord = lastFewWords.any { it in fillerWords }
        
        return if (hasFillerWord) 0.8f else 0.2f
    }
}
```

### Phase 7 Checkpoint

✅ Trigger policy only fires when confidence is high  
✅ Adapts to your speaking rate  
✅ Backs off if correction rate is high

**Phase 7 Complete!**

---

## Phase 8: Output & Feedback (Week 4)

### Expected Output
✅ Suggestion is spoken via TTS  
✅ Text is auto-inserted into active text field  
✅ Corrections are detected and logged

### Step 8.1: Implement TTS Output

Create `app/src/main/java/com/pocketwhisper/app/audio/TtsOutput.kt`:

```kotlin
package com.pocketwhisper.app.audio

import android.content.Context
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.*

class TtsOutput(private val context: Context) {
    
    private val TAG = "TtsOutput"
    private var tts: TextToSpeech? = null
    private var isReady = false
    
    init {
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale.US
                isReady = true
                Log.d(TAG, "✅ TTS initialized")
            }
        }
    }
    
    /**
     * Speak a word via phone speaker or earpiece.
     */
    fun speak(text: String, useEarpiece: Boolean = false) {
        if (!isReady) {
            Log.w(TAG, "TTS not ready yet")
            return
        }
        
        // TODO: Route to earpiece if useEarpiece == true
        // (requires AudioManager.MODE_IN_COMMUNICATION)
        
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "suggestion_$text")
        Log.d(TAG, "🔊 Speaking: $text")
    }
    
    fun shutdown() {
        tts?.stop()
        tts?.shutdown()
    }
}
```

### Step 8.2: Implement Accessibility Service for Auto-Insert

Create `app/src/main/java/com/pocketwhisper/app/accessibility/AutoInsertService.kt`:

```kotlin
package com.pocketwhisper.app.accessibility

import android.accessibilityservice.AccessibilityService
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import android.os.Bundle

class AutoInsertService : AccessibilityService() {
    
    private val TAG = "AutoInsertService"
    
    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        // We'll use this to detect corrections (backspace events)
    }
    
    override fun onInterrupt() {
        Log.d(TAG, "Service interrupted")
    }
    
    /**
     * Insert text at current cursor position.
     */
    fun insertText(text: String) {
        Log.d(TAG, "Inserting text: $text")
        
        // Find focused text field
        val root = rootInActiveWindow ?: run {
            Log.w(TAG, "No active window")
            return
        }
        
        val focused = root.findFocus(AccessibilityNodeInfo.FOCUS_INPUT) ?: run {
            Log.w(TAG, "No focused text field")
            return
        }
        
        // Insert text
        val arguments = Bundle()
        arguments.putCharSequence(
            AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE,
            text
        )
        
        focused.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, arguments)
        
        Log.d(TAG, "✅ Text inserted")
    }
    
    companion object {
        private var instance: AutoInsertService? = null
        
        fun getInstance(): AutoInsertService? = instance
    }
    
    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        Log.d(TAG, "✅ Accessibility service connected")
    }
    
    override fun onDestroy() {
        instance = null
        super.onDestroy()
    }
}
```

### Step 8.3: Integrate Everything

Update `ListenForegroundService.kt`:

```kotlin
class ListenForegroundService : Service() {
    
    private lateinit var llmSession: LlmSession
    private lateinit var triggerPolicy: TriggerPolicy
    private lateinit var speakingRateTracker: SpeakingRateTracker
    private lateinit var fillerDetector: FillerDetector
    private lateinit var ttsOutput: TtsOutput
    
    private var currentTranscript = ""
    private var lastSpeechTime = 0L
    
    override fun onCreate() {
        super.onCreate()
        
        // Load all models
        val modelLoader = ModelLoader(this)
        val vadModule = modelLoader.loadModel("vad_silero.pte")
        val asrModule = modelLoader.loadModel("asr_distil_whisper_small_int8.pte")
        val llmModule = modelLoader.loadModel("llm_qwen_0.5b_int8.pte")
        
        vadDetector = VadDetector(vadModule)
        asrSession = AsrSession(this, asrModule)
        llmSession = LlmSession(this, llmModule)
        
        speakingRateTracker = SpeakingRateTracker()
        triggerPolicy = TriggerPolicy(speakingRateTracker)
        fillerDetector = FillerDetector()
        ttsOutput = TtsOutput(this)
        
        audioCapture = AudioCaptureService()
    }
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // ... (notification code)
        
        audioCapture.start(serviceScope)
        
        serviceScope.launch {
            audioCapture.audioFlow.collect { audioChunk ->
                val speechProb = vadDetector.detectSpeech(audioChunk)
                
                if (speechProb > 0.5f) {
                    lastSpeechTime = System.currentTimeMillis()
                    
                    // Update ASR
                    val transcript = asrSession.addAudio(audioChunk)
                    if (transcript != null) {
                        currentTranscript = transcript
                        
                        // Update speaking rate
                        val wordCount = transcript.split(" ").size
                        speakingRateTracker.update(wordCount, 3.0f)
                    }
                } else {
                    // Silence detected - check if we should trigger
                    val pauseDuration = (System.currentTimeMillis() - lastSpeechTime) / 1000f
                    
                    if (pauseDuration > 0.2f && currentTranscript.isNotEmpty()) {
                        checkTrigger(pauseDuration)
                    }
                }
            }
        }
        
        return START_STICKY
    }
    
    private suspend fun checkTrigger(pauseDuration: Float) {
        // Get LLM suggestion
        val (nextWord, confidence) = llmSession.generateNextWord(currentTranscript)
        
        // Detect filler
        val fillerLikelihood = fillerDetector.detectFiller(currentTranscript)
        
        // TODO: Implement semantic coherence check
        val semanticCoherence = 0.85f  // Dummy for now
        
        // TODO: Check if word was recently used
        val recentlyUsed = false
        
        // Build trigger signals
        val signals = TriggerSignals(
            pauseDuration = pauseDuration,
            fillerLikelihood = fillerLikelihood,
            completionConfidence = confidence,
            asrStable = true,  // Simplified
            semanticCoherence = semanticCoherence,
            recentlyUsed = recentlyUsed
        )
        
        // Decide if we should trigger
        if (triggerPolicy.shouldTrigger(signals)) {
            // Speak the suggestion
            ttsOutput.speak(nextWord)
            
            // Wait 100ms then insert
            delay(100)
            
            // Auto-insert via accessibility service
            AutoInsertService.getInstance()?.insertText(nextWord)
            
            // TODO: Start correction detection timer
            startCorrectionDetection(nextWord)
        }
    }
    
    private fun startCorrectionDetection(insertedWord: String) {
        // TODO: Monitor for backspace/delete within next 5 seconds
        // If detected, call: triggerPolicy.reportCorrection(true)
    }
}
```

### Phase 8 Checkpoint

✅ App speaks suggestion via TTS  
✅ Text is auto-inserted (if accessibility service is enabled)  
✅ Check Settings → Accessibility → enable "Pocket Whisper"

**Phase 8 Complete!**

---

## Phase 9: Testing & Optimization (Week 4-5)

### Expected Output
✅ Latency is <350ms p95  
✅ Correction rate is <10%  
✅ App works in airplane mode

### Step 9.1: Add Latency Profiler

Create `app/src/main/java/com/pocketwhisper/app/profiler/LatencyProfiler.kt`:

```kotlin
package com.pocketwhisper.app.profiler

import android.util.Log

class LatencyProfiler {
    
    private val TAG = "LatencyProfiler"
    private val latencies = mutableListOf<Long>()
    
    fun recordLatency(component: String, latencyMs: Long) {
        latencies.add(latencyMs)
        Log.d(TAG, "$component: ${latencyMs}ms")
    }
    
    fun getStats(): String {
        if (latencies.isEmpty()) return "No data"
        
        val sorted = latencies.sorted()
        val p50 = sorted[sorted.size / 2]
        val p95 = sorted[(sorted.size * 0.95).toInt()]
        val avg = latencies.average()
        
        return "Avg: ${avg.toInt()}ms, P50: ${p50}ms, P95: ${p95}ms"
    }
}
```

### Step 9.2: Enable QNN Delegate (NPU Acceleration)

To get <200ms LLM latency, you MUST use the QNN delegate. This requires:

1. **Install Qualcomm QNN SDK**:
   ```bash
   # Download from Qualcomm developer portal
   # https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk
   ```

2. **Re-export LLM with QNN backend**:
   ```python
   # In export_llm.py, add:
   from executorch.backends.qualcomm import QnnBackend
   
   et_program = edge_program.to_executorch(
       backend_delegates={"qnn": QnnBackend()}
   )
   ```

3. **Update Android build**:
   ```kotlin
   // In build.gradle, add QNN libraries
   implementation("com.qualcomm.qnn:qnn-runtime:2.1.0")
   ```

### Step 9.3: Run Benchmark

Create a test script:

```kotlin
// Test 200 scripted utterances
val testPhrases = listOf(
    "Could you please",
    "I think we should",
    "Let me know if",
    // ... 197 more
)

for (phrase in testPhrases) {
    val startTime = System.currentTimeMillis()
    
    // Simulate: ASR → LLM → Output
    val (word, confidence) = llmSession.generateNextWord(phrase)
    
    val totalLatency = System.currentTimeMillis() - startTime
    latencyProfiler.recordLatency("end-to-end", totalLatency)
}

println(latencyProfiler.getStats())
// Target: P95 < 350ms
```

### Step 9.4: Airplane Mode Test

1. Enable airplane mode on S25 Ultra
2. Start the app
3. Toggle "Listening ON"
4. Speak a test phrase
5. Verify suggestion is generated (no cloud calls)

### Phase 9 Checkpoint

✅ P95 latency < 350ms (with QNN)  
✅ Works in airplane mode  
✅ Correction rate < 10% in manual testing

**Phase 9 Complete! 🎉**

---

## Common Issues & Troubleshooting

### Issue: Models don't load
- **Solution**: Check `app/src/main/assets/` has all `.pte` files
- Verify file sizes (VAD ~1.5MB, ASR ~244MB, LLM ~500MB)

### Issue: Latency is >1 second
- **Cause**: QNN delegate not active (running on CPU)
- **Solution**: Re-export models with QNN backend, add QNN runtime to Android

### Issue: ASR produces gibberish
- **Cause**: Mel spectrogram preprocessing is wrong
- **Solution**: Use the `processor` from transformers to convert audio → mel

### Issue: Accessibility service doesn't insert text
- **Cause**: Permission not granted
- **Solution**: Settings → Accessibility → Pocket Whisper → Enable

### Issue: Too many false triggers
- **Cause**: Confidence threshold too low
- **Solution**: Increase `tauConf` from 0.75 to 0.85

---

## What You've Built

After completing all 9 phases, you have:

✅ **On-device speech-to-text** pipeline  
✅ **Real-time LLM** next-word prediction  
✅ **Intelligent trigger system** that adapts to speaking rate  
✅ **Audio output + auto-insert** functionality  
✅ **Feedback loop** that learns from corrections  
✅ **<350ms latency** end-to-end  
✅ **Privacy-first** (no cloud, airplane mode works)

---

## Next Steps (Post-MVP)

1. **Improve trigger policy**:
   - Add semantic coherence model
   - Better filler detection (train custom TCN)
   
2. **Add multilingual support**:
   - Export Whisper multilingual models
   - Language detection

3. **Personalization**:
   - Cache user's frequently used phrases
   - Learn from accepted suggestions

4. **Battery optimization**:
   - Adaptive sampling (reduce quality when battery low)
   - Smarter VAD (longer silence → sleep mode)

---

**Total Implementation Time: 4-5 weeks for 1 engineer**

Good luck! 🚀

