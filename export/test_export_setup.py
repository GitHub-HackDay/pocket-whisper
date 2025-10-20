#!/usr/bin/env python3
"""
Quick test to verify the export environment is set up correctly
Tests with Silero VAD since it's the smallest model
"""

import sys
import importlib
from pathlib import Path

def check_dependency(package_name: str, import_name: str = None) -> bool:
    """Check if a package is installed and can be imported."""
    import_name = import_name or package_name
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def main():
    print("🔍 Checking Pocket Whisper Export Environment")
    print("=" * 50)
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python version: {python_version}")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    print("✅ Python version OK\n")
    
    # Check required packages
    dependencies = [
        ("torch", "torch", "PyTorch"),
        ("torchaudio", "torchaudio", "TorchAudio"),
        ("transformers", "transformers", "Transformers"),
        ("librosa", "librosa", "Librosa"),
        ("numpy", "numpy", "NumPy"),
    ]
    
    optional_dependencies = [
        ("executorch", "executorch", "ExecuTorch"),
        ("torchao", "torchao", "TorchAO (Quantization)"),
    ]
    
    print("Required Dependencies:")
    all_required_ok = True
    for package, import_name, display_name in dependencies:
        if check_dependency(package, import_name):
            print(f"  ✅ {display_name}")
        else:
            print(f"  ❌ {display_name} - Run: pip install {package}")
            all_required_ok = False
    
    print("\nOptional Dependencies:")
    for package, import_name, display_name in optional_dependencies:
        if check_dependency(package, import_name):
            print(f"  ✅ {display_name}")
        else:
            print(f"  ⚠️  {display_name} - Optional, models will use fallback")
    
    if not all_required_ok:
        print("\n❌ Missing required dependencies. Run:")
        print("   ./setup_export_env.sh")
        return False
    
    # Test PyTorch and check for available backends
    print("\n📊 PyTorch Configuration:")
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
    
    # Test simple model export
    print("\n🧪 Testing simple model export...")
    try:
        # Create a tiny test model
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TinyModel()
        model.eval()
        
        # Test input
        x = torch.randn(1, 10)
        
        # Test forward pass
        with torch.no_grad():
            output = model(x)
        print(f"  ✅ Model forward pass OK (output shape: {output.shape})")
        
        # Test tracing
        traced = torch.jit.trace(model, x)
        print("  ✅ TorchScript tracing OK")
        
        # Test quantization if available
        if check_dependency("torchao"):
            from torch.ao.quantization import quantize_dynamic
            quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            print("  ✅ Int8 quantization OK")
        else:
            print("  ⚠️  Quantization not tested (torchao not installed)")
        
        # Test ExecuTorch export if available
        if check_dependency("executorch"):
            try:
                from executorch.exir import to_edge
                import torch.export as torch_export
                
                edge_program = to_edge(
                    torch_export.export(model, (x,))
                )
                print("  ✅ ExecuTorch export OK")
            except Exception as e:
                print(f"  ⚠️  ExecuTorch export failed: {e}")
        else:
            print("  ⚠️  ExecuTorch not available - will use TorchScript fallback")
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False
    
    # Check output directory
    print("\n📁 Checking output directory...")
    assets_dir = Path("../app/src/main/assets")
    if not assets_dir.exists():
        print(f"  📝 Creating assets directory: {assets_dir}")
        assets_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"  ✅ Assets directory exists: {assets_dir}")
    
    # Summary
    print("\n" + "=" * 50)
    print("✅ Environment is ready for model export!")
    print("\nNext steps:")
    print("  1. Run individual model exports:")
    print("     python export_vad.py")
    print("     python export_asr.py")
    print("     python export_llm.py")
    print("\n  2. Or export all at once:")
    print("     python export_all.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
