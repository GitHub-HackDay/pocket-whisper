#!/usr/bin/env python3
"""Test the exported VAD model to ensure it works correctly."""

import torch
import numpy as np
import sys
from pathlib import Path

def test_vad_model():
    """Test the exported VAD model with simulated audio."""
    
    model_path = Path("../app/src/main/assets/vad_model.pt")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    print(f"Loading model from {model_path}")
    print(f"Model size: {model_path.stat().st_size / 1024:.1f} KB")
    
    # Load the TorchScript model
    try:
        model = torch.jit.load(model_path, map_location='cpu')
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Check model structure
    print("\n📋 Model Info:")
    print(f"Type: {type(model)}")
    try:
        print(f"Methods: {[m for m in dir(model) if not m.startswith('_')][:10]}")
    except:
        pass
    
    # Test inference with different input shapes
    print("\n🧪 Testing inference...")
    
    # Test 1: Standard input (1, 512)
    print("\nTest 1: Standard input shape [1, 512]")
    try:
        audio_chunk = torch.randn(1, 512)
        print(f"Input shape: {audio_chunk.shape}")
        print(f"Input dtype: {audio_chunk.dtype}")
        
        with torch.no_grad():
            output = model(audio_chunk)
        
        print(f"✅ Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output value: {output.item():.4f}")
        
        # Check if output is in valid range
        if 0 <= output.item() <= 1:
            print("✅ Output is in valid probability range [0, 1]")
        else:
            print(f"⚠️ Output {output.item()} is outside probability range")
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
    
    # Test 2: Try with batch dimension (512,)
    print("\nTest 2: Without batch dimension [512]")
    try:
        audio_chunk = torch.randn(512)
        print(f"Input shape: {audio_chunk.shape}")
        
        with torch.no_grad():
            output = model(audio_chunk)
        
        print(f"✅ Output shape: {output.shape}")
        print(f"Output value: {output.item():.4f}")
    except Exception as e:
        print(f"❌ Test 2 failed (expected): {e}")
        print("  Model requires batch dimension")
    
    # Test 3: Speech-like signal
    print("\nTest 3: Speech-like signal")
    try:
        # Create speech-like signal (higher amplitude, more variation)
        t = torch.linspace(0, 0.032, 512)  # 32ms at 16kHz
        speech_signal = (
            0.3 * torch.sin(2 * np.pi * 200 * t) +  # 200Hz component
            0.2 * torch.sin(2 * np.pi * 500 * t) +  # 500Hz component
            0.1 * torch.randn(512)  # noise
        )
        speech_signal = speech_signal.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            output = model(speech_signal)
        
        print(f"✅ Speech probability: {output.item():.4f}")
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
    
    # Test 4: Silence/noise signal
    print("\nTest 4: Silence/noise signal")
    try:
        # Create silence/noise signal (low amplitude)
        silence_signal = 0.01 * torch.randn(1, 512)
        
        with torch.no_grad():
            output = model(silence_signal)
        
        print(f"✅ Silence probability: {output.item():.4f}")
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
    
    # Test 5: Check model graph (if possible)
    print("\n📊 Model Structure:")
    try:
        # Try to print the graph
        print(model.graph)
    except:
        try:
            # Try to print the code
            print(model.code)
        except:
            print("Could not access model graph/code")
    
    print("\n✅ VAD model testing complete!")
    print("\n💡 Important notes for Android integration:")
    print("1. Input must be shape [1, 512] (batch_size=1, samples=512)")
    print("2. Input should be Float32 tensor")
    print("3. Output is a single float probability [0, 1]")
    print("4. Higher values indicate speech, lower values indicate silence")

if __name__ == "__main__":
    test_vad_model()
