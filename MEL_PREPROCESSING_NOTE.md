# Mel Preprocessing Implementation Note

## Decision: Implement in Pure Kotlin

After attempting ONNX export, we've determined that **mel spectrogram preprocessing should be implemented directly in Kotlin** as originally planned.

### Why Kotlin Implementation:

1. **ONNX Limitations**: STFT operation in ONNX has complex type limitations
2. **Simple Math**: Mel preprocessing is straightforward DSP (framing, windowing, FFT, mel filters)
3. **Better Control**: Direct implementation gives us full control over performance
4. **No Extra Dependency**: One less model file to load

### Implementation Path:

Use **JTransforms** library for FFT (already a standard Android library):

```kotlin
// In MelPreprocessorKotlin.kt
class MelPreprocessorKotlin {
    fun audioToMel(audio: FloatArray): Array<FloatArray> {
        // 1. Frame audio into 25ms windows with 10ms stride
        // 2. Apply Hann window
        // 3. Compute FFT (using JTransforms)
        // 4. Compute power spectrogram
        // 5. Apply mel filterbank (80 filters)
        // 6. Convert to log scale
        // 7. Normalize (mean=0, std=1)
        return melSpectrogram
    }
}
```

### Full code is in `IMPLEMENTATION_GUIDE.md` starting at line 798

This is the **recommended approach** and aligns with the full implementation plan!

