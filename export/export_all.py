#!/usr/bin/env python3
"""
Master export script for Pocket Whisper models
Exports all three models to ExecuTorch format and verifies outputs
"""

import sys
import subprocess
from pathlib import Path
import time

def run_export(script_name: str) -> bool:
    """Run an export script and return success status."""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print('='*60)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully")
            return True
        else:
            print(f"❌ {script_name} failed with error:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Failed to run {script_name}: {e}")
        return False


def verify_exports():
    """Verify all exported files exist and have reasonable sizes."""
    
    print("\n" + "="*60)
    print("Verifying Exported Models")
    print("="*60)
    
    assets_dir = Path("../app/src/main/assets")
    
    expected_files = [
        # VAD model (either .pte or .pt)
        ("vad_silero_v4.pte", 1.0, 3.0),  # Expected 1-3 MB
        ("vad_silero_v4.pt", 1.0, 3.0),   # Fallback TorchScript
        
        # ASR model (either .pte or .pt)
        ("asr_wav2vec2_base_int8.pte", 50.0, 150.0),  # Expected 50-150 MB
        ("asr_wav2vec2_base_int8.pt", 50.0, 150.0),   # Fallback
        
        # LLM model (either .pte or .pt)
        ("llm_qwen2_0.5b_int8.pte", 300.0, 600.0),    # Expected 300-600 MB
        ("llm_qwen2_0.5b_int8_qnn.pte", 300.0, 600.0), # QNN version
        ("llm_qwen2_0.5b_int8.pt", 300.0, 600.0),     # Fallback
    ]
    
    found_models = {
        "vad": False,
        "asr": False, 
        "llm": False
    }
    
    print("\n📁 Checking exported files:")
    
    for filename, min_size_mb, max_size_mb in expected_files:
        file_path = assets_dir / filename
        
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            
            # Determine model type
            if "vad" in filename:
                model_type = "vad"
            elif "asr" in filename or "wav2vec2" in filename:
                model_type = "asr"
            elif "llm" in filename or "qwen" in filename:
                model_type = "llm"
            else:
                model_type = "unknown"
            
            if min_size_mb <= size_mb <= max_size_mb:
                print(f"  ✅ {filename}: {size_mb:.2f} MB")
                found_models[model_type] = True
            else:
                print(f"  ⚠️  {filename}: {size_mb:.2f} MB (expected {min_size_mb}-{max_size_mb} MB)")
    
    # Check processor/tokenizer directories
    processor_dirs = [
        ("asr_processor", "ASR processor"),
        ("llm_tokenizer", "LLM tokenizer")
    ]
    
    print("\n📁 Checking processor/tokenizer files:")
    for dir_name, description in processor_dirs:
        dir_path = assets_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            num_files = len(list(dir_path.iterdir()))
            print(f"  ✅ {description}: {num_files} files")
        else:
            print(f"  ❌ {description}: Not found")
    
    # Summary
    print("\n" + "="*60)
    print("Export Summary")
    print("="*60)
    
    all_found = all(found_models.values())
    
    if all_found:
        print("✅ All models exported successfully!")
        print("\nModel Status:")
        print(f"  • VAD (Silero v4): {'✅ Ready' if found_models['vad'] else '❌ Missing'}")
        print(f"  • ASR (Wav2Vec2-base): {'✅ Ready' if found_models['asr'] else '❌ Missing'}")
        print(f"  • LLM (Qwen2-0.5B): {'✅ Ready' if found_models['llm'] else '❌ Missing'}")
        
        # Calculate total size
        total_size_mb = sum(
            (assets_dir / f).stat().st_size / (1024 * 1024)
            for f in assets_dir.iterdir()
            if f.is_file() and (f.suffix in ['.pte', '.pt'])
        )
        print(f"\n📊 Total model size: {total_size_mb:.2f} MB")
        
        if total_size_mb > 800:
            print("  ⚠️  Models are quite large. Consider further quantization for production.")
        
        return True
    else:
        print("❌ Some models are missing:")
        for model_type, found in found_models.items():
            if not found:
                print(f"  • {model_type.upper()} model not found")
        return False


def main():
    """Main export pipeline."""
    
    print("🚀 Pocket Whisper Model Export Pipeline")
    print("="*60)
    print("Models to export:")
    print("  1. Silero VAD v4 - Voice Activity Detection")
    print("  2. Wav2Vec2-base - Automatic Speech Recognition") 
    print("  3. Qwen2-0.5B - Next-word Prediction LLM")
    print("="*60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    # Export each model
    export_scripts = [
        "export_vad.py",
        "export_asr.py", 
        "export_llm.py"
    ]
    
    results = []
    for script in export_scripts:
        success = run_export(script)
        results.append(success)
        
        if not success:
            print(f"\n⚠️  {script} failed, but continuing with other exports...")
    
    # Verify all exports
    verification_success = verify_exports()
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    if all(results) and verification_success:
        print("🎉 All models exported and verified successfully!")
        print("\nNext steps:")
        print("  1. cd ../")
        print("  2. ./build_and_deploy.sh")
        print("  3. Test the app on your S25 Ultra")
    else:
        print("⚠️  Some exports failed or are incomplete.")
        print("\nFailed exports:")
        for i, (script, success) in enumerate(zip(export_scripts, results)):
            if not success:
                print(f"  • {script}")
        
        print("\nTroubleshooting:")
        print("  1. Check Python dependencies: pip install -r requirements-export.txt")
        print("  2. Ensure you have enough disk space (need ~2GB free)")
        print("  3. Check individual export script logs above")
        print("  4. Try running failed scripts individually")
        
    return all(results) and verification_success


if __name__ == "__main__":
    start_time = time.time()
    success = main()
    elapsed_time = time.time() - start_time
    
    print(f"\n⏱️  Total export time: {elapsed_time/60:.1f} minutes")
    
    sys.exit(0 if success else 1)
