#!/usr/bin/env python3
"""
Export Wav2Vec2-base model for ASR to ExecuTorch format (.pte)
Model: facebook/wav2vec2-base-960h
Size: ~360MB (fp32) -> ~90MB (int8 quantized)
Latency target: 80-120ms per 1-second chunk

Why Wav2Vec2 over Whisper:
- Direct CTC decoding (no autoregressive decoder complexity)
- Better streaming support
- Lower latency for real-time applications
- Simpler post-processing
"""

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from pathlib import Path
import numpy as np
import sys
import time
from torch.ao.quantization import quantize_dynamic
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def load_wav2vec2_model():
    """Load Wav2Vec2-base-960h model and processor."""
    
    model_name = "facebook/wav2vec2-base-960h"
    print(f"📥 Loading {model_name}...")
    
    # Load processor (handles audio preprocessing)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    
    # Load model
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Start with fp32, will quantize later
    )
    
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"   Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    print(f"   Vocab size: {model.config.vocab_size}")
    
    return model, processor


def export_wav2vec2_to_executorch():
    """Export Wav2Vec2 to ExecuTorch format with int8 quantization."""
    
    # 1. Load model and processor
    model, processor = load_wav2vec2_model()
    
    # 2. Prepare sample input
    # Wav2Vec2 expects audio at 16kHz
    sample_rate = 16000
    duration = 1.0  # 1 second chunk for streaming
    num_samples = int(sample_rate * duration)
    
    # Create sample audio input
    sample_audio = torch.randn(num_samples)
    print(f"\n📊 Sample audio shape: {sample_audio.shape}")
    
    # Process audio to get model input
    inputs = processor(
        sample_audio.numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True
    )
    
    input_values = inputs.input_values
    print(f"   Processed input shape: {input_values.shape}")
    
    # 3. Test inference
    print("\n🧪 Testing model inference...")
    start_time = time.time()
    with torch.no_grad():
        logits = model(input_values).logits
    inference_time = (time.time() - start_time) * 1000
    
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Inference time (fp32): {inference_time:.2f}ms")
    
    # Decode to text (for testing)
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    print(f"   Test transcription: '{transcription}'")
    
    # 4. Quantize model to int8
    print("\n⚡ Quantizing model to int8...")
    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Linear},  # Quantize Linear layers
        dtype=torch.qint8
    )
    
    # Test quantized inference
    start_time = time.time()
    with torch.no_grad():
        q_logits = quantized_model(input_values).logits
    q_inference_time = (time.time() - start_time) * 1000
    print(f"   Quantized inference time: {q_inference_time:.2f}ms")
    print(f"   Speedup: {inference_time / q_inference_time:.2f}x")
    
    # 5. Export to TorchScript (intermediate step)
    print("\n📝 Tracing model for export...")
    traced_model = torch.jit.trace(quantized_model, input_values)
    
    # Save TorchScript model
    torchscript_path = Path("../app/src/main/assets/asr_wav2vec2_base_int8.pt")
    torchscript_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced_model, str(torchscript_path))
    print(f"   TorchScript saved to: {torchscript_path}")
    
    # 6. Export to ExecuTorch
    print("\n🚀 Converting to ExecuTorch format...")
    try:
        from executorch.exir import to_edge, EdgeCompileConfig
        import torch.export as torch_export
        
        # Export to Edge format
        edge_program = to_edge(
            torch_export.export(
                quantized_model,
                (input_values,),
                dynamic_shapes=None
            )
        )
        
        # Compile to ExecuTorch
        et_program = edge_program.to_executorch(
            config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True  # Skip dimension order checks for Wav2Vec2
            )
        )
        
        # Save .pte file
        output_path = Path("../app/src/main/assets/asr_wav2vec2_base_int8.pte")
        with open(output_path, "wb") as f:
            f.write(et_program.buffer)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n✅ Successfully exported to ExecuTorch!")
        print(f"   File: {output_path}")
        print(f"   Size: {file_size_mb:.2f} MB")
        
    except ImportError as e:
        print(f"⚠️  ExecuTorch not fully available, keeping TorchScript version")
        print(f"   Error: {e}")
        output_path = torchscript_path
    
    # 7. Save processor configuration
    print("\n💾 Saving processor configuration...")
    processor_dir = Path("../app/src/main/assets/asr_processor")
    processor_dir.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(str(processor_dir))
    
    # Also save a simplified vocab for Android
    vocab = processor.tokenizer.get_vocab()
    vocab_path = processor_dir / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"   Processor saved to: {processor_dir}")
    print(f"   Vocab size: {len(vocab)} tokens")
    
    # 8. Create streaming ASR wrapper
    streaming_wrapper = '''
"""
Streaming ASR wrapper for Wav2Vec2
Processes audio in overlapping chunks for continuous transcription
"""

import torch
import numpy as np
from collections import deque
from typing import Optional, List, Tuple

class StreamingASR:
    def __init__(self, model_path: str, processor):
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.processor = processor
        
        # Streaming buffer (3 seconds of context)
        self.sample_rate = 16000
        self.chunk_duration = 1.0  # Process 1 second at a time
        self.context_duration = 3.0
        
        self.audio_buffer = deque(maxlen=int(self.sample_rate * self.context_duration))
        self.last_transcript = ""
        
    def add_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[str]:
        """Add audio chunk and return partial transcript if available."""
        
        # Add to buffer
        self.audio_buffer.extend(audio_chunk)
        
        # Process if we have enough audio (at least 1 second)
        if len(self.audio_buffer) >= self.sample_rate * self.chunk_duration:
            return self._process_buffer()
        
        return None
    
    def _process_buffer(self) -> str:
        """Process current audio buffer and return transcript."""
        
        # Convert buffer to array
        audio_array = np.array(self.audio_buffer)
        
        # Process with Wav2Vec2 processor
        inputs = self.processor(
            audio_array,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Run inference
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
        
        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = self.processor.batch_decode(predicted_ids)[0]
        
        # Simple deduplication for streaming
        if transcript != self.last_transcript:
            self.last_transcript = transcript
            return transcript
        
        return None
    
    def reset(self):
        """Reset the streaming buffer."""
        self.audio_buffer.clear()
        self.last_transcript = ""

# Usage example:
# streamer = StreamingASR("model.pt", processor)
# for audio_chunk in audio_stream:
#     transcript = streamer.add_audio_chunk(audio_chunk)
#     if transcript:
#         print(f"Transcript: {transcript}")
'''
    
    wrapper_path = Path("streaming_asr.py")
    wrapper_path.write_text(streaming_wrapper)
    print(f"\n📝 Streaming wrapper saved to: {wrapper_path}")
    
    return output_path


if __name__ == "__main__":
    print("=" * 60)
    print("Wav2Vec2-base Export to ExecuTorch")
    print("=" * 60)
    
    try:
        output_file = export_wav2vec2_to_executorch()
        print("\n🎉 ASR export completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)