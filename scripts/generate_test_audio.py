#!/usr/bin/env python3
"""
Generate test audio files for development and testing.
Creates sample WAV files at 16kHz mono for use with Pocket Whisper.
"""

import numpy as np
import soundfile as sf
from pathlib import Path


def create_test_audio_files():
    """Generate test audio files for different scenarios."""

    # Create test_audio directory
    output_dir = Path("test_audio")
    output_dir.mkdir(exist_ok=True)

    print("Generating test audio files...")

    # Configuration
    sample_rate = 16000  # 16kHz (standard for speech models)

    # 1. Silence (2 seconds)
    print("  [1/3] Creating silence.wav (2s)")
    silence = np.zeros(sample_rate * 2, dtype=np.float32)
    sf.write(output_dir / "silence.wav", silence, sample_rate)

    # 2. Speech simulation - sine wave pattern (3 seconds)
    # This simulates speech-like patterns for testing
    print("  [2/3] Creating speech.wav (3s)")
    duration = 3
    t = np.linspace(0, duration, sample_rate * duration)

    # Simulate speech with modulated sine waves (fundamental + harmonics)
    fundamental = 200  # Hz (typical male voice)
    speech = (
        0.3 * np.sin(2 * np.pi * fundamental * t)
        + 0.2 * np.sin(2 * np.pi * fundamental * 2 * t)
        + 0.1 * np.sin(2 * np.pi * fundamental * 3 * t)
    )

    # Add amplitude modulation to simulate words
    modulation = np.abs(np.sin(2 * np.pi * 3 * t))  # 3 "words" per second
    speech = speech * modulation

    # Add some noise
    noise = np.random.normal(0, 0.02, len(speech))
    speech = speech + noise

    # Normalize
    speech = speech / np.max(np.abs(speech)) * 0.8
    speech = speech.astype(np.float32)

    sf.write(output_dir / "speech.wav", speech, sample_rate)

    # 3. Filler simulation (2 seconds with pauses)
    print("  [3/3] Creating filler.wav (2s)")
    duration = 2
    t = np.linspace(0, duration, sample_rate * duration)

    # Create a pattern: "uh" (0.3s) + pause (0.2s) + "I think" (0.5s) + pause (0.3s) + "we should" (0.7s)
    filler = np.zeros(len(t), dtype=np.float32)

    # "uh" - lower pitch, short
    uh_samples = int(0.3 * sample_rate)
    uh = 0.3 * np.sin(2 * np.pi * 150 * np.linspace(0, 0.3, uh_samples))
    filler[:uh_samples] = uh

    # "I think" - normal speech pattern
    think_start = int(0.5 * sample_rate)
    think_samples = int(0.5 * sample_rate)
    think = 0.4 * np.sin(2 * np.pi * 200 * np.linspace(0, 0.5, think_samples))
    think *= np.abs(
        np.sin(2 * np.pi * 4 * np.linspace(0, 0.5, think_samples))
    )  # 2 words
    filler[think_start : think_start + think_samples] = think

    # "we should" - normal speech pattern
    should_start = int(1.3 * sample_rate)
    should_samples = int(0.7 * sample_rate)
    should = 0.4 * np.sin(2 * np.pi * 200 * np.linspace(0, 0.7, should_samples))
    should *= np.abs(
        np.sin(2 * np.pi * 3 * np.linspace(0, 0.7, should_samples))
    )  # 2 words
    filler[should_start : should_start + should_samples] = should

    # Add noise
    noise = np.random.normal(0, 0.02, len(filler))
    filler = filler + noise

    # Normalize
    filler = filler / np.max(np.abs(filler)) * 0.8

    sf.write(output_dir / "filler.wav", filler, sample_rate)

    print("\n✅ Test audio files created successfully!")
    print(f"   Output directory: {output_dir.absolute()}")
    print(f"   Files created:")
    print(f"     - silence.wav (2s, 16kHz mono)")
    print(f"     - speech.wav (3s, 16kHz mono)")
    print(f"     - filler.wav (2s, 16kHz mono)")


if __name__ == "__main__":
    create_test_audio_files()
