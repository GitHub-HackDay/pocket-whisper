#!/usr/bin/env python3
"""
Export Wav2Vec2-base to TorchScript for PyTorch Mobile
Simplified for hackathon - prioritizes working over perfect
"""

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pathlib import Path
import time

def export_asr_for_mobile():
    """Export Wav2Vec2 to TorchScript (.pt) for PyTorch Mobile."""
    
    print("="*60)
    print("🚀 Exporting Wav2Vec2-base for PyTorch Mobile")
    print("="*60)
    
    model_name = "facebook/wav2vec2-base-960h"
    
    # 1. Load model
    print(f"\n📥 Loading {model_name}...")
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model.eval()
    
    print(f"✅ Model loaded")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    # 2. Prepare sample input
    print("\n🧪 Testing model...")
    sample_rate = 16000
    duration = 1.0  # 1 second
    test_audio = torch.randn(int(sample_rate * duration))
    
    # Process audio
    inputs = processor(test_audio.numpy(), sampling_rate=sample_rate, return_tensors="pt")
    input_values = inputs.input_values
    
    # Test inference
    start = time.time()
    with torch.no_grad():
        logits = model(input_values).logits
    latency = (time.time() - start) * 1000
    
    print(f"   Input shape: {input_values.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Latency: {latency:.2f}ms")
    
    # 3. Trace model
    print("\n📝 Tracing model for mobile...")
    traced_model = torch.jit.trace(model, input_values)
    
    # 4. Optimize for mobile
    print("⚡ Optimizing for mobile...")
    optimized = torch.jit.optimize_for_mobile(traced_model)
    
    # 5. Save model
    output_path = Path("../app/src/main/assets/asr_wav2vec2_base.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    optimized._save_for_lite_interpreter(str(output_path))
    
    # 6. Save processor config (needed for tokenization)
    processor_path = Path("../app/src/main/assets/asr_processor")
    processor_path.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(str(processor_path))
    
    size_mb = output_path.stat().st_size / (1024*1024)
    print(f"\n✅ SUCCESS!")
    print(f"   Model: {output_path}")
    print(f"   Size: {size_mb:.1f} MB")
    print(f"   Processor: {processor_path}")
    print(f"   Format: TorchScript (PyTorch Mobile)")
    
    # 7. Usage example
    print("\n📱 Android Usage:")
    print("""
// Load in Kotlin
val module = Module.load(assetFilePath("asr_wav2vec2_base.pt"))
val input = Tensor.fromBlob(audioData, longArrayOf(1, audioLength))
val logits = module.forward(IValue.from(input)).toTensor()
// Decode logits to text using processor vocab
""")
    
    return output_path

if __name__ == "__main__":
    export_asr_for_mobile()
