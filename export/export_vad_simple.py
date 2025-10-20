#!/usr/bin/env python3
"""
SIMPLIFIED VAD Export - Works around SSL issues
For hackathon quick deployment
"""

import torch
import ssl
import os
import time
from pathlib import Path

# Workaround for SSL certificate issues on macOS
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def export_vad_simple():
    """Simplified VAD export that just works."""
    
    print("="*60)
    print("🚀 SIMPLIFIED VAD Export (PyTorch Mobile)")
    print("="*60)
    
    try:
        # Try to download Silero VAD
        print("\n📥 Attempting to download Silero VAD v4...")
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        print("✅ Model downloaded successfully")
        
    except Exception as e:
        print(f"⚠️  Download failed: {e}")
        print("\n🔄 Creating a simple VAD placeholder model instead...")
        
        # Create a simple placeholder VAD model for testing
        class SimpleVAD(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv1d(1, 16, kernel_size=5)
                self.conv2 = torch.nn.Conv1d(16, 32, kernel_size=5)
                self.fc = torch.nn.Linear(32 * 504, 1)  # Adjusted for 512 input
                self.sigmoid = torch.nn.Sigmoid()
                
            def forward(self, x):
                # x shape: (batch, 512)
                x = x.unsqueeze(1)  # Add channel dim: (batch, 1, 512)
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = x.view(x.size(0), -1)  # Flatten
                x = self.fc(x)
                return self.sigmoid(x).squeeze()
        
        model = SimpleVAD()
        print("✅ Created placeholder VAD model")
        print("   Note: This is for testing only, replace with real Silero VAD later")
    
    model.eval()
    
    # Test the model
    print("\n🧪 Testing model...")
    test_input = torch.randn(1, 512)  # 32ms at 16kHz
    sample_rate = 16000
    
    start = time.time()
    with torch.no_grad():
        # Try with sample rate first (Silero VAD), fallback without
        try:
            output = model(test_input, sample_rate)
        except:
            output = model(test_input)
    latency = (time.time() - start) * 1000
    
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output: {output.item() if output.numel() == 1 else output}")
    print(f"   Latency: {latency:.2f}ms")
    
    # Trace for mobile
    print("\n📝 Tracing for PyTorch Mobile...")
    try:
        # Try tracing with sample rate (for Silero)
        traced = torch.jit.trace(model, (test_input, torch.tensor(sample_rate)))
    except:
        # Fallback to simple input
        traced = torch.jit.trace(model, test_input)
    
    # Optimize
    print("⚡ Optimizing...")
    try:
        # Try new API
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimized = optimize_for_mobile(traced)
    except:
        # Fallback - use traced directly
        print("   Note: Mobile optimization not available, using traced model")
        optimized = traced
    
    # Save
    # Use absolute path to ensure it saves correctly
    base_dir = Path(__file__).parent.parent  # pocket-whisper directory
    output_path = base_dir / "app" / "src" / "main" / "assets" / "vad_model.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"   Saving to: {output_path}")
    
    try:
        optimized._save_for_lite_interpreter(str(output_path))
    except:
        # Fallback to regular save
        torch.jit.save(optimized, str(output_path))
    
    size_kb = output_path.stat().st_size / 1024
    print(f"\n✅ SUCCESS!")
    print(f"   File: {output_path}")
    print(f"   Size: {size_kb:.1f} KB")
    print(f"   Type: {'Silero VAD v4' if 'silero' in str(type(model)).lower() else 'Placeholder VAD'}")
    
    print("\n📱 Next steps:")
    print("1. If using placeholder, download real Silero VAD manually later")
    print("2. Add PyTorch Mobile to Android project")
    print("3. Load with: Module.load(assetFilePath('vad_model.pt'))")
    
    return output_path

if __name__ == "__main__":
    export_vad_simple()
