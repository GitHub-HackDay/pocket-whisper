#!/usr/bin/env python3
"""
Verify all exported models are present and correctly formatted.
"""

from pathlib import Path
import sys


def verify_exports():
    """Check that all required models and configs are exported."""
    
    print("=" * 70)
    print("VERIFYING MODEL EXPORTS")
    print("=" * 70)
    
    assets_dir = Path("../app/src/main/assets")
    
    required_files = {
        "VAD Model": assets_dir / "vad_silero.onnx",
        "ASR Encoder": assets_dir / "asr_distil_whisper_small_int8.pte",
        "ASR Processor": assets_dir / "asr_processor" / "preprocessor_config.json",
        "ASR Tokenizer": assets_dir / "asr_tokenizer" / "tokenizer_config.json",
        "LLM Model": assets_dir / "llm_qwen_0.5b_mobile.ptl",
        "LLM Tokenizer": assets_dir / "llm_tokenizer" / "tokenizer_config.json",
    }
    
    expected_sizes = {
        "VAD Model": (2, 3),  # MB range
        "ASR Encoder": (330, 350),
        "LLM Model": (1800, 1900),
    }
    
    print("\nChecking files...")
    all_good = True
    
    for name, filepath in required_files.items():
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            
            # Check size if it's a model file
            if name in expected_sizes:
                min_size, max_size = expected_sizes[name]
                if min_size <= size_mb <= max_size:
                    print(f"  ✅ {name}: {size_mb:.1f} MB")
                else:
                    print(f"  ⚠️  {name}: {size_mb:.1f} MB (expected {min_size}-{max_size} MB)")
                    all_good = False
            else:
                print(f"  ✅ {name}: Found")
        else:
            print(f"  ❌ {name}: MISSING at {filepath}")
            all_good = False
    
    # Check for optional QNN models
    print("\nChecking optional QNN models...")
    qnn_files = {
        "ASR QNN": assets_dir / "asr_distil_whisper_small_qnn.pte",
        "LLM QNN": assets_dir / "llm_qwen_0.5b_qnn.pte",
    }
    
    for name, filepath in qnn_files.items():
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  🚀 {name}: {size_mb:.1f} MB (NPU acceleration available!)")
        else:
            print(f"  ℹ️  {name}: Not found (CPU fallback will be used)")
    
    # Summary
    print("\n" + "=" * 70)
    if all_good:
        print("✅ ALL REQUIRED MODELS PRESENT AND VALID")
        print("=" * 70)
        print("\nTotal model size:")
        
        total_size = 0
        for name, filepath in required_files.items():
            if filepath.exists() and filepath.is_file():
                total_size += filepath.stat().st_size / (1024 * 1024)
        
        print(f"  ~{total_size:.0f} MB on disk")
        print("\nReady for Android integration!")
        return 0
    else:
        print("❌ SOME FILES MISSING OR INVALID")
        print("=" * 70)
        print("\nRun export scripts:")
        print("  python export/export_vad.py")
        print("  python export/export_asr.py")
        print("  python export/export_llm_torchscript.py")
        return 1


if __name__ == "__main__":
    sys.exit(verify_exports())

