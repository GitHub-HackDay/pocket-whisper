#!/usr/bin/env python3
"""
Export Distil-Whisper Small model for streaming ASR.
This model converts audio to text transcription.
"""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def export_distil_whisper():
    """Export Distil-Whisper Small model for streaming ASR."""

    print("=" * 70)
    print("EXPORTING DISTIL-WHISPER SMALL MODEL")
    print("=" * 70)

    model_id = "distil-whisper/distil-small.en"

    # Load model and processor
    print(f"\n[1/7] Loading {model_id}...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    print("  ✓ Model and processor loaded")

    # Save processor config (needed for mel spectrogram preprocessing)
    print("\n[2/7] Saving processor configuration...")
    processor_path = Path("../app/src/main/assets/asr_processor/")
    processor_path.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(processor_path)
    print(f"  ✓ Processor saved to: {processor_path}")

    # Export encoder only (decoder will be handled separately)
    print("\n[3/7] Extracting encoder...")
    base_encoder = model.get_encoder()
    base_encoder.eval()

    # Wrap encoder to return plain tensor instead of BaseModelOutput
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_features):
            # Get encoder output and extract just the last_hidden_state tensor
            output = self.encoder(input_features)
            return output.last_hidden_state  # Return tensor, not BaseModelOutput

    encoder = EncoderWrapper(base_encoder)
    encoder.eval()
    print("  ✓ Encoder extracted and wrapped")

    # Process sample to get mel spec shape
    print("\n[4/7] Processing sample audio...")
    sample_audio = torch.randn(1, 16000 * 3)  # 3 seconds at 16kHz
    with torch.no_grad():
        inputs = processor(
            sample_audio.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
        )
        input_features = inputs.input_features  # Shape: (1, 80, time)

    print(f"  ✓ Input shape: {input_features.shape}")

    # Export encoder with proper settings
    print("\n[5/7] Exporting encoder...")
    with torch.no_grad():
        # Use strict=False to avoid dynamic shape constraints
        exported = torch.export.export(encoder, (input_features,), strict=False)
    print("  ✓ Encoder exported")

    # Convert to ExecuTorch (skip quantization for now - will optimize later)
    print("\n[6/7] Converting to ExecuTorch...")
    from executorch.exir import to_edge, EdgeCompileConfig

    edge_program = to_edge(
        exported, compile_config=EdgeCompileConfig(_check_ir_validity=False)
    )
    et_program = edge_program.to_executorch()
    print("  ✓ Converted to ExecuTorch")
    print("  ℹ Note: Quantization will be added in optimization phase")

    # Save
    print("\n[7/7] Saving .pte file...")
    output_path = Path("../app/src/main/assets/asr_distil_whisper_small_int8.pte")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved to: {output_path}")
    print(f"  ✓ File size: {size_mb:.1f} MB")

    # Save tokenizer separately
    print("\nSaving tokenizer...")
    tokenizer = processor.tokenizer
    tokenizer_path = Path("../app/src/main/assets/asr_tokenizer/")
    tokenizer_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    print(f"  ✓ Tokenizer saved to: {tokenizer_path}")

    # VALIDATE
    print("\nValidating export...")
    validate_asr_export(output_path, input_features)

    print("\n" + "=" * 70)
    print("✅ ASR MODEL EXPORT COMPLETE!")
    print("=" * 70)
    print(f"\nEncoder output: {output_path.absolute()}")
    print(f"Size: {size_mb:.1f} MB (~244 MB expected)")
    print(f"Processor: {processor_path.absolute()}")
    print(f"Tokenizer: {tokenizer_path.absolute()}")
    print("\nNext steps:")
    print(
        "  1. Implement mel spectrogram preprocessing (see export_mel_preprocessor.py)"
    )
    print("  2. Implement tokenizer in Kotlin (WhisperTokenizer.kt)")
    print("  3. Test loading in Android")


def validate_asr_export(pte_path, sample_input):
    """Test exported ASR encoder."""
    from executorch.extension.pybindings import portable_lib

    try:
        runtime = portable_lib._load_for_executorch(str(pte_path))

        # Test forward pass
        output = runtime.forward([sample_input])[0]
        # Output should be (batch, time, hidden_dim) = (1, time_steps, 384)
        print(f"  ✓ ASR encoder output shape: {output.shape}")
        print(f"  ✓ Expected format: (batch, time, 384)")
        print("  ✓ Validation successful!")

    except Exception as e:
        print(f"  ✗ Validation failed: {e}")
        raise


if __name__ == "__main__":
    export_distil_whisper()
