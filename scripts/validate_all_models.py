#!/usr/bin/env python3
"""
Validate all exported models with test audio.
Run this after exporting models to ensure they work correctly.
"""

import torch
import torchaudio
from pathlib import Path
import sys


def validate_all():
    """Run all model validations with test audio."""

    print("=" * 70)
    print("VALIDATING ALL EXPORTED MODELS")
    print("=" * 70)

    errors = []

    # Check if test audio exists
    audio_path = Path("test_audio/speech.wav")
    if not audio_path.exists():
        print("\n❌ Test audio not found!")
        print("   Please run: python scripts/generate_test_audio.py")
        sys.exit(1)

    # Load test audio
    print("\n[SETUP] Loading test audio...")
    audio, sr = torchaudio.load(audio_path)
    if sr != 16000:
        print(f"  ⚠ Resampling from {sr}Hz to 16kHz")
        audio = torchaudio.functional.resample(audio, sr, 16000)
        sr = 16000

    print(f"  ✓ Loaded: {audio_path}")
    print(f"  ✓ Shape: {audio.shape}, Sample rate: {sr}Hz")

    # Test VAD
    print("\n" + "=" * 70)
    print("[1/3] TESTING VAD MODEL")
    print("=" * 70)

    vad_path = Path("app/src/main/assets/vad_silero.pte")
    if not vad_path.exists():
        errors.append("VAD model not found")
        print(f"  ✗ Model not found: {vad_path}")
        print("     Run: python export/export_vad.py")
    else:
        try:
            from executorch.extension.pybindings import portable_lib

            vad = portable_lib._load_for_executorch(str(vad_path))
            print(f"  ✓ Model loaded: {vad_path}")

            # Test on multiple chunks
            chunks = audio.squeeze().split(512)
            print(f"\n  Testing on {len(chunks)} chunks (512 samples each):")

            speech_detections = 0
            for i, chunk in enumerate(chunks[:10]):  # Test first 10 chunks
                if chunk.shape[0] == 512:
                    chunk_tensor = chunk.unsqueeze(0)  # Add batch dimension
                    prob = vad.forward([chunk_tensor])[0]

                    if prob > 0.5:
                        speech_detections += 1

                    status = "🗣️ SPEECH" if prob > 0.5 else "🤫 SILENCE"
                    print(f"    Chunk {i:2d}: {prob:.3f} - {status}")

            print(f"\n  ✓ Speech detected in {speech_detections}/10 chunks")
            print("  ✓ VAD validation passed!")

        except Exception as e:
            errors.append(f"VAD validation failed: {e}")
            print(f"  ✗ VAD validation failed: {e}")

    # Test ASR encoder
    print("\n" + "=" * 70)
    print("[2/3] TESTING ASR MODEL")
    print("=" * 70)

    asr_path = Path("app/src/main/assets/asr_distil_whisper_small_int8.pte")
    processor_path = Path("app/src/main/assets/asr_processor")

    if not asr_path.exists():
        errors.append("ASR model not found")
        print(f"  ✗ Model not found: {asr_path}")
        print("     Run: python export/export_asr.py")
    elif not processor_path.exists():
        errors.append("ASR processor config not found")
        print(f"  ✗ Processor config not found: {processor_path}")
        print("     Run: python export/export_asr.py")
    else:
        try:
            from executorch.extension.pybindings import portable_lib
            from transformers import AutoProcessor

            # Load processor
            processor = AutoProcessor.from_pretrained(processor_path)
            print(f"  ✓ Processor loaded: {processor_path}")

            # Preprocess audio
            inputs = processor(
                audio.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
            )
            input_features = inputs.input_features
            print(f"  ✓ Audio preprocessed to shape: {input_features.shape}")

            # Load and run model
            asr = portable_lib._load_for_executorch(str(asr_path))
            print(f"  ✓ Model loaded: {asr_path}")

            output = asr.forward([input_features])[0]
            print(f"  ✓ Encoder output shape: {output.shape}")
            print("  ✓ ASR validation passed!")
            print("\n  ℹ Note: Full transcription requires decoder implementation")

        except Exception as e:
            errors.append(f"ASR validation failed: {e}")
            print(f"  ✗ ASR validation failed: {e}")

    # Test LLM
    print("\n" + "=" * 70)
    print("[3/3] TESTING LLM MODEL")
    print("=" * 70)

    llm_path = Path("app/src/main/assets/llm_qwen_0.5b_int8_cpu.pte")
    tokenizer_path = Path("app/src/main/assets/llm_tokenizer")

    if not llm_path.exists():
        errors.append("LLM model not found")
        print(f"  ✗ Model not found: {llm_path}")
        print("     Run: python export/export_llm.py")
    elif not tokenizer_path.exists():
        errors.append("LLM tokenizer config not found")
        print(f"  ✗ Tokenizer config not found: {tokenizer_path}")
        print("     Run: python export/export_llm.py")
    else:
        try:
            from executorch.extension.pybindings import portable_lib
            from transformers import AutoTokenizer

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            print(f"  ✓ Tokenizer loaded: {tokenizer_path}")

            # Tokenize sample text
            sample_text = "Could you please"
            inputs = tokenizer(sample_text, return_tensors="pt")
            input_ids = inputs.input_ids
            print(f"  ✓ Sample text tokenized: '{sample_text}'")
            print(f"    Token IDs: {input_ids.tolist()[0]}")

            # Load and run model
            llm = portable_lib._load_for_executorch(str(llm_path))
            print(f"  ✓ Model loaded: {llm_path}")

            output = llm.forward([input_ids])[0]
            print(f"  ✓ LLM output shape: {output.shape}")

            # Get next token prediction
            logits = output[0, -1, :]  # Last position logits
            next_token_id = torch.argmax(torch.from_numpy(logits)).item()
            next_token = tokenizer.decode([next_token_id])

            print(f"  ✓ Next token prediction: '{next_token}'")
            print("  ✓ LLM validation passed!")

        except Exception as e:
            errors.append(f"LLM validation failed: {e}")
            print(f"  ✗ LLM validation failed: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    if errors:
        print(f"\n❌ {len(errors)} ERROR(S) FOUND:\n")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease fix the errors above before proceeding.")
        sys.exit(1)
    else:
        print("\n✅ ALL MODELS VALIDATED SUCCESSFULLY!")
        print("\nYou can now proceed to Android integration:")
        print("  1. Copy all .pte files to Android app/src/main/assets/")
        print("  2. Copy processor and tokenizer configs")
        print("  3. Implement ModelLoader.kt and test loading on device")


if __name__ == "__main__":
    validate_all()
