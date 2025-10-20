#!/usr/bin/env python3
"""
Simple ASR export script with progress tracking
"""

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pathlib import Path
import time
import sys

def export_asr_simple():
    print("="*60)
    print("🚀 Exporting Wav2Vec2 ASR Model")
    print("="*60)
    
    model_name = "facebook/wav2vec2-base-960h"
    
    try:
        # Load model with force_download to avoid cache issues
        print(f"\n📥 Loading {model_name}...")
        print("   This may take 5-10 minutes for first download (~360MB)")
        
        model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            force_download=False,  # Set to True if you want to re-download
            cache_dir=None,  # Use default cache
            local_files_only=False  # Allow downloading if not cached
        )
        processor = Wav2Vec2Processor.from_pretrained(
            model_name,
            force_download=False
        )
        
        model.eval()
        print(f"✅ Model loaded successfully")
        
        # Create dummy input
        print("\n🧪 Creating test input...")
        sample_rate = 16000
        test_audio = torch.randn(1, sample_rate)  # 1 second of audio
        
        # Test forward pass
        print("🔬 Testing model...")
        with torch.no_grad():
            output = model(test_audio)
            logits = output.logits
        print(f"   Output shape: {logits.shape}")
        
        # Create a wrapper to simplify output
        class SimpleWav2Vec2(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                # Return only the logits, not the full ModelOutput
                return self.model(x).logits
        
        # Wrap the model
        simple_model = SimpleWav2Vec2(model)
        simple_model.eval()
        
        # Trace the simplified model
        print("\n📝 Tracing model for mobile...")
        with torch.no_grad():
            traced_model = torch.jit.trace(simple_model, test_audio)
        
        # Optimize for mobile
        print("⚡ Optimizing for mobile...")
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimized = optimize_for_mobile(traced_model)
        
        # Save model
        output_dir = Path("../app/src/main/assets")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "asr_wav2vec2_base.pt"
        print(f"\n💾 Saving to {output_path}...")
        optimized._save_for_lite_interpreter(str(output_path))
        
        # Save processor
        processor_path = output_dir / "asr_processor"
        processor_path.mkdir(parents=True, exist_ok=True)
        processor.save_pretrained(str(processor_path))
        
        # Report success
        size_mb = output_path.stat().st_size / (1024*1024)
        print(f"\n✅ SUCCESS!")
        print(f"   Model saved: {output_path}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Processor saved: {processor_path}")
        
        return str(output_path)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Try setting force_download=True to re-download")
        print("3. Check if you have enough disk space")
        sys.exit(1)

if __name__ == "__main__":
    export_asr_simple()
