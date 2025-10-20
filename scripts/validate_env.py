#!/usr/bin/env python3
"""
Validation script for Python environment and ExecuTorch installation.
This script tests that all required packages are installed and working correctly.
"""

import sys


def validate_environment():
    """Test all required packages and their core functionality."""
    print("=" * 60)
    print("Pocket Whisper - Environment Validation")
    print("=" * 60)

    errors = []

    # Test 1: PyTorch
    print("\n[1/8] Testing PyTorch...")
    try:
        import torch

        print(f"  ✓ PyTorch {torch.__version__} imported successfully")

        # Test basic tensor operations
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = x @ y
        assert z.shape == (3, 3), "Matrix multiplication failed"
        print(f"  ✓ Basic tensor operations work")
    except Exception as e:
        errors.append(f"PyTorch: {e}")
        print(f"  ✗ PyTorch error: {e}")

    # Test 2: TorchAudio
    print("\n[2/8] Testing TorchAudio...")
    try:
        import torchaudio

        print(f"  ✓ TorchAudio {torchaudio.__version__} imported successfully")
    except Exception as e:
        errors.append(f"TorchAudio: {e}")
        print(f"  ✗ TorchAudio error: {e}")

    # Test 3: Transformers
    print("\n[3/8] Testing Transformers...")
    try:
        import transformers

        print(f"  ✓ Transformers {transformers.__version__} imported successfully")
    except Exception as e:
        errors.append(f"Transformers: {e}")
        print(f"  ✗ Transformers error: {e}")

    # Test 4: ExecuTorch
    print("\n[4/8] Testing ExecuTorch...")
    try:
        import executorch
        from executorch.exir import to_edge

        print(f"  ✓ ExecuTorch imported successfully")
        print(f"  ✓ to_edge function available")
    except Exception as e:
        errors.append(f"ExecuTorch: {e}")
        print(f"  ✗ ExecuTorch error: {e}")

    # Test 5: TorchAO
    print("\n[5/8] Testing TorchAO...")
    try:
        import torchao

        print(f"  ✓ TorchAO imported successfully")
    except Exception as e:
        errors.append(f"TorchAO: {e}")
        print(f"  ✗ TorchAO error: {e}")

    # Test 6: Librosa
    print("\n[6/8] Testing Librosa...")
    try:
        import librosa

        print(f"  ✓ Librosa {librosa.__version__} imported successfully")
    except Exception as e:
        errors.append(f"Librosa: {e}")
        print(f"  ✗ Librosa error: {e}")

    # Test 7: SoundFile
    print("\n[7/8] Testing SoundFile...")
    try:
        import soundfile

        print(f"  ✓ SoundFile {soundfile.__version__} imported successfully")
    except Exception as e:
        errors.append(f"SoundFile: {e}")
        print(f"  ✗ SoundFile error: {e}")

    # Test 8: ONNX
    print("\n[8/8] Testing ONNX...")
    try:
        import onnx

        print(f"  ✓ ONNX {onnx.__version__} imported successfully")
    except Exception as e:
        errors.append(f"ONNX: {e}")
        print(f"  ✗ ONNX error: {e}")

    # Test torch.export
    print("\n[BONUS] Testing torch.export...")
    try:
        model = torch.nn.Linear(10, 5)
        sample_input = torch.randn(1, 10)
        exported = torch.export.export(model, (sample_input,))
        print(f"  ✓ torch.export.export works")

        # Test ExecuTorch conversion
        from executorch.exir import to_edge

        edge_program = to_edge(exported)
        et_program = edge_program.to_executorch()
        print(f"  ✓ ExecuTorch conversion works")
        print(f"  ✓ .pte buffer size: {len(et_program.buffer)} bytes")
    except Exception as e:
        errors.append(f"torch.export: {e}")
        print(f"  ✗ torch.export error: {e}")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"❌ VALIDATION FAILED - {len(errors)} error(s)")
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease fix the errors above before proceeding.")
        sys.exit(1)
    else:
        print("✅ ALL TESTS PASSED!")
        print("Environment is ready for Pocket Whisper development.")
    print("=" * 60)


if __name__ == "__main__":
    validate_environment()
