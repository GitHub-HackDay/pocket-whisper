#!/usr/bin/env python3
"""
Export Silero VAD v4 model to ExecuTorch format (.pte)
Model size: ~1.5MB
Latency target: <5ms per 32ms frame
"""

import torch
import torchaudio
from pathlib import Path
import sys
import time
from typing import Tuple

# Add parent directory to path for utils
sys.path.append(str(Path(__file__).parent.parent))


def download_silero_vad_v4() -> Tuple[torch.nn.Module, dict]:
    """Download and load Silero VAD v4 model."""
    print("📥 Downloading Silero VAD v4...")
    
    # Load the latest Silero VAD v4 model
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False,
        trust_repo=True
    )
    
    # Get utility functions
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    
    print(f"✅ Silero VAD loaded successfully")
    print(f"   Model type: {type(model)}")
    
    return model, utils


def export_silero_vad_to_executorch():
    """Export Silero VAD to ExecuTorch format with optimizations."""
    
    # 1. Load model
    model, utils = download_silero_vad_v4()
    model.eval()
    
    # 2. Prepare sample input
    # Silero VAD expects 512 samples at 16kHz (32ms window)
    batch_size = 1
    chunk_size = 512  # 32ms at 16kHz
    sample_rate = 16000
    
    # Create sample input tensor
    sample_input = torch.randn(batch_size, chunk_size)
    print(f"📊 Sample input shape: {sample_input.shape}")
    
    # 3. Test inference
    print("🧪 Testing model inference...")
    start_time = time.time()
    with torch.no_grad():
        output = model(sample_input, sample_rate)
    inference_time = (time.time() - start_time) * 1000
    print(f"   Output shape: {output.shape}")
    print(f"   Inference time: {inference_time:.2f}ms")
    print(f"   Output (speech probability): {output.item():.4f}")
    
    # 4. Trace the model
    print("\n📝 Tracing model for export...")
    traced_model = torch.jit.trace(model, (sample_input, torch.tensor(sample_rate)))
    
    # 5. Export to TorchScript first (intermediate step)
    print("💾 Saving TorchScript model...")
    torchscript_path = Path("../app/src/main/assets/vad_silero_v4.pt")
    torchscript_path.parent.mkdir(parents=True, exist_ok=True)
    traced_model.save(str(torchscript_path))
    
    # 6. Export to ExecuTorch
    print("\n🚀 Converting to ExecuTorch format...")
    try:
        from executorch.exir import to_edge
        from executorch.exir import EdgeCompileConfig
        import torch.export as torch_export
        
        # Export to Edge format
        edge_program = to_edge(
            torch_export.export(
                model,
                (sample_input,),  # Just audio input for simpler model
                dynamic_shapes=None
            )
        )
        
        # Compile to ExecuTorch
        et_program = edge_program.to_executorch(
            config=EdgeCompileConfig(_check_ir_validity=False)
        )
        
        # Save .pte file
        output_path = Path("../app/src/main/assets/vad_silero_v4.pte")
        with open(output_path, "wb") as f:
            f.write(et_program.buffer)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n✅ Successfully exported to ExecuTorch!")
        print(f"   File: {output_path}")
        print(f"   Size: {file_size_mb:.2f} MB")
        
    except ImportError as e:
        print(f"⚠️  ExecuTorch not available, using TorchScript fallback")
        print(f"   Error: {e}")
        print(f"   TorchScript model saved at: {torchscript_path}")
        return torchscript_path
    
    # 7. Create test script
    print("\n📝 Creating test script...")
    test_script = '''
import torch
import numpy as np

def test_vad_model(model_path):
    """Test VAD model with sample audio."""
    
    # Load model (TorchScript for now)
    model = torch.jit.load(model_path)
    model.eval()
    
    # Create test audio: silence -> speech -> silence
    sample_rate = 16000
    duration = 3  # seconds
    t = np.linspace(0, duration, sample_rate * duration)
    
    # Generate synthetic "speech" (more complex waveform)
    speech = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz tone
    speech += np.sin(2 * np.pi * 880 * t) * 0.3  # Harmonic
    speech += np.random.randn(len(t)) * 0.1  # Add noise
    
    # Add silence padding
    silence = np.zeros(sample_rate // 2)  # 0.5s silence
    audio = np.concatenate([silence, speech, silence])
    
    # Process in chunks
    chunk_size = 512  # 32ms at 16kHz
    chunks = [audio[i:i+chunk_size] for i in range(0, len(audio)-chunk_size, chunk_size)]
    
    results = []
    for i, chunk in enumerate(chunks):
        chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0)
        with torch.no_grad():
            prob = model(chunk_tensor, torch.tensor(sample_rate))
        results.append(prob.item())
        
        if i % 10 == 0:
            print(f"Chunk {i:3d}: Speech prob = {prob.item():.3f}")
    
    return results

if __name__ == "__main__":
    test_vad_model("../app/src/main/assets/vad_silero_v4.pt")
'''
    
    test_path = Path("test_vad.py")
    test_path.write_text(test_script)
    print(f"   Test script saved to: {test_path}")
    
    return output_path


if __name__ == "__main__":
    print("=" * 60)
    print("Silero VAD v4 Export to ExecuTorch")
    print("=" * 60)
    
    try:
        output_file = export_silero_vad_to_executorch()
        print("\n🎉 Export completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)