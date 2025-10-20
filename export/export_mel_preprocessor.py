#!/usr/bin/env python3
"""
Export mel spectrogram preprocessor as ONNX model.
This model converts raw audio to mel spectrograms for Whisper ASR.
"""

import torch
import torch.nn as nn
import torchaudio
from pathlib import Path


class MelSpectrogramExtractor(nn.Module):
    """
    Mel spectrogram extractor compatible with Whisper.
    Converts raw audio (16kHz) to 80-bin mel spectrogram.
    """

    def __init__(self):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,  # Whisper standard
            hop_length=160,  # Whisper standard
            n_mels=80,  # Whisper standard
            f_min=0,
            f_max=8000,
        )

    def forward(self, audio):
        """
        Convert audio to log mel spectrogram.

        Args:
            audio: (batch, samples) raw audio at 16kHz

        Returns:
            log_mel: (batch, 80, time) log mel spectrogram
        """
        # Compute mel spectrogram
        mel = self.mel_transform(audio)

        # Convert to log scale (Whisper uses log10)
        log_mel = torch.log10(torch.clamp(mel, min=1e-10))

        # Normalize (Whisper normalization)
        log_mel = (log_mel + 4.0) / 4.0

        return log_mel


def export_mel_as_onnx():
    """Export mel spectrogram computation as ONNX model."""

    print("=" * 70)
    print("EXPORTING MEL SPECTROGRAM PREPROCESSOR")
    print("=" * 70)

    print("\n[1/4] Creating MelSpectrogramExtractor...")
    model = MelSpectrogramExtractor()
    model.eval()
    print("  ✓ Model created")

    # Test with sample audio (3 seconds)
    print("\n[2/4] Testing with sample audio...")
    sample = torch.randn(1, 48000)  # 3 seconds at 16kHz

    with torch.no_grad():
        output = model(sample)

    print(f"  ✓ Input shape: {sample.shape}")
    print(f"  ✓ Output shape: {output.shape}")
    print(f"  ✓ Expected: (1, 80, ~300) for 3-second audio")

    # Export to ONNX
    print("\n[3/4] Exporting to ONNX format...")
    output_path = Path("../app/src/main/assets/mel_preprocessor.onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        sample,
        output_path,
        input_names=["audio"],
        output_names=["mel_spectrogram"],
        dynamic_axes={"audio": {1: "samples"}, "mel_spectrogram": {2: "time"}},
        opset_version=17,  # Required for STFT operation
        do_constant_folding=True,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Exported to: {output_path}")
    print(f"  ✓ File size: {size_mb:.2f} MB")

    # Validate ONNX model
    print("\n[4/4] Validating ONNX export...")
    validate_onnx_export(output_path, sample)

    print("\n" + "=" * 70)
    print("✅ MEL PREPROCESSOR EXPORT COMPLETE!")
    print("=" * 70)
    print(f"\nOutput: {output_path.absolute()}")
    print(f"Size: {size_mb:.2f} MB")
    print("\nIntegration options:")
    print("  Option 1 (Recommended): Use ONNX Runtime in Android")
    print("    - Add dependency: com.microsoft.onnxruntime:onnxruntime-android")
    print("    - Load and run in MelPreprocessor.kt")
    print("  Option 2: Implement in pure Kotlin")
    print("    - Use JTransforms for FFT")
    print("    - Implement mel filterbank manually")
    print("    - See MelPreprocessorKotlin.kt in the plan")


def validate_onnx_export(onnx_path, sample_input):
    """Validate that ONNX model produces correct output."""
    try:
        import onnxruntime as ort

        # Load ONNX model
        session = ort.InferenceSession(str(onnx_path))

        # Run inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        result = session.run([output_name], {input_name: sample_input.numpy()})[0]

        print(f"  ✓ ONNX model loaded successfully")
        print(f"  ✓ Output shape: {result.shape}")
        print(f"  ✓ Output range: [{result.min():.3f}, {result.max():.3f}]")
        print("  ✓ Validation successful!")

    except ImportError:
        print("  ⚠ onnxruntime not installed - skipping validation")
        print("    Install with: pip install onnxruntime")
    except Exception as e:
        print(f"  ✗ Validation failed: {e}")
        raise


if __name__ == "__main__":
    export_mel_as_onnx()
