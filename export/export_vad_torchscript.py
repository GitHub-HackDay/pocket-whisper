#!/usr/bin/env python3
"""
Export Silero VAD v4 to TorchScript for PyTorch Mobile
Simpler, more reliable path for hackathon
"""

import torch
import time
from pathlib import Path

def export_vad_for_mobile():
    """Export Silero VAD to TorchScript (.pt) for PyTorch Mobile."""
    
    print("="*60)
    print("🚀 Exporting Silero VAD v4 for PyTorch Mobile")
    print("="*60)
    
    # 1. Download Silero VAD
    print("\n📥 Downloading Silero VAD v4...")
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False,
        trust_repo=True
    )
    
    model.eval()
    print("✅ Model downloaded")
    
    # 2. Test inference
    print("\n🧪 Testing model...")
    sample_rate = 16000
    chunk_size = 512  # 32ms at 16kHz
    test_audio = torch.randn(1, chunk_size)
    
    start = time.time()
    with torch.no_grad():
        output = model(test_audio, sample_rate)
    latency = (time.time() - start) * 1000
    
    print(f"   Output (speech prob): {output.item():.4f}")
    print(f"   Latency: {latency:.2f}ms")
    
    # 3. Trace for mobile
    print("\n📝 Tracing model for mobile...")
    traced_model = torch.jit.trace(model, test_audio)
    
    # 4. Optimize for mobile
    print("⚡ Optimizing for mobile...")
    traced_model = torch.jit.optimize_for_mobile(traced_model)
    
    # 5. Save
    output_path = Path("../app/src/main/assets/vad_silero_v4.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced_model._save_for_lite_interpreter(str(output_path))
    
    size_mb = output_path.stat().st_size / (1024*1024)
    print(f"\n✅ SUCCESS!")
    print(f"   File: {output_path}")
    print(f"   Size: {size_mb:.2f} MB")
    print(f"   Format: TorchScript (PyTorch Mobile)")
    
    return output_path

if __name__ == "__main__":
    export_vad_for_mobile()
