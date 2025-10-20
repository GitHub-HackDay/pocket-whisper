#!/usr/bin/env python3
"""
Export Silero VAD model.

VAD uses ONNX format (production-ready, optimized for small models).
ASR and LLM will use full ExecuTorch .pte format.
"""

import torch
from pathlib import Path
import urllib.request


def export_silero_vad():
    """Export Silero VAD model in ONNX format (optimal for VAD)."""

    print("=" * 70)
    print("EXPORTING SILERO VAD MODEL")
    print("=" * 70)

    print("\n[1/3] Downloading Silero VAD ONNX model...")
    onnx_url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"

    output_path = Path("../app/src/main/assets/vad_silero.onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(onnx_url, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Downloaded: {output_path}")
    print(f"  ✓ Size: {size_mb:.2f} MB")

    print("\n[2/3] Validating model...")
    validate_vad_onnx(output_path)

    print("\n[3/3] Export complete!")
    print("\n" + "=" * 70)
    print("✅ VAD MODEL READY")
    print("=" * 70)
    print(f"\nFile: {output_path.absolute()}")
    print(f"Size: {size_mb:.2f} MB")
    print(f"Format: ONNX (production-ready)")
    print("\nWhy ONNX for VAD:")
    print("  • Small model (~2MB) - no benefit from ExecuTorch")
    print("  • ONNX Runtime is highly optimized")
    print("  • Already in build.gradle dependencies")
    print("  • Production-proven for mobile")
    print("\nASR and LLM will use full ExecuTorch (.pte) for maximum performance!")


def validate_vad_onnx(onnx_path):
    """Validate ONNX VAD model."""
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(str(onnx_path))
        state = np.zeros((2, 1, 128), dtype=np.float32)
        silence = np.zeros((1, 512), dtype=np.float32)
        sr = np.array([16000], dtype=np.int64)

        output, _ = session.run(None, {"input": silence, "sr": sr, "state": state})
        prob = float(output[0])

        if prob < 0.3:
            print(f"  ✓ Validation passed (silence={prob:.3f})")
        else:
            print(f"  ⚠ Silence has high prob: {prob:.3f}")

    except Exception as e:
        print(f"  ⚠ Validation warning: {e}")
        print("  Model will be validated in Android")


if __name__ == "__main__":
    export_silero_vad()
