package com.pocketwhisper.app.ml

import android.util.Log
import kotlin.math.*

/**
 * Mel Spectrogram processor for Distil-Whisper preprocessing.
 * 
 * Converts raw audio to 80-bin mel spectrograms using:
 * - STFT with 400 sample (25ms) window and 160 sample (10ms) hop
 * - 80 mel filterbanks from 0 to 8000 Hz
 * - Log scaling
 * 
 * Matches Whisper's preprocessing: 
 * https://github.com/openai/whisper/blob/main/whisper/audio.py
 */
class MelSpectrogramProcessor {
    
    private val TAG = "MelSpectrogramProcessor"
    
    // Whisper constants
    private val SAMPLE_RATE = 16000
    private val N_FFT = 400 // FFT window size (25ms)
    private val HOP_LENGTH = 160 // Hop size (10ms)
    private val N_MELS = 80 // Number of mel bins
    private val MEL_MIN_FREQ = 0.0
    private val MEL_MAX_FREQ = 8000.0
    
    // Pre-computed Hann window
    private val hannWindow = FloatArray(N_FFT) { i ->
        (0.5 * (1 - cos(2 * PI * i / N_FFT))).toFloat()
    }
    
    // Pre-computed mel filterbank
    private val melFilterbank = createMelFilterbank()
    
    /**
     * Convert audio to mel spectrogram.
     * 
     * @param audio Raw audio samples (16kHz, mono)
     * @return Mel spectrogram (80 x time_steps)
     */
    fun audioToMel(audio: FloatArray): Array<FloatArray> {
        // 1. Compute STFT
        val stft = computeSTFT(audio)
        
        // 2. Convert to power spectrogram |X|^2
        val powerSpec = stftToPower(stft)
        
        // 3. Apply mel filterbank
        val melSpec = applyMelFilterbank(powerSpec)
        
        // 4. Log scaling: log10(max(1e-10, x))
        return melSpec.map { mel ->
            mel.map { value ->
                log10(max(1e-10f, value))
            }.toFloatArray()
        }.toTypedArray()
    }
    
    /**
     * Compute Short-Time Fourier Transform (STFT).
     * 
     * @param audio Input audio
     * @return Complex STFT (time_steps x (N_FFT/2 + 1) x 2) where [:,:,0] = real, [:,:,1] = imag
     */
    private fun computeSTFT(audio: FloatArray): Array<Array<FloatArray>> {
        // Pad audio to ensure we have enough samples
        val paddedAudio = audio + FloatArray((N_FFT - audio.size % N_FFT) % N_FFT) { 0f }
        
        // Number of frames
        val numFrames = ((paddedAudio.size - N_FFT) / HOP_LENGTH) + 1
        val freqBins = N_FFT / 2 + 1
        
        // Output: [numFrames][freqBins][2] (2 = real, imag)
        val stft = Array(numFrames) { Array(freqBins) { FloatArray(2) } }
        
        // Process each frame
        for (frameIdx in 0 until numFrames) {
            val start = frameIdx * HOP_LENGTH
            
            // Extract frame and apply Hann window
            val frame = FloatArray(N_FFT) { i ->
                if (start + i < paddedAudio.size) {
                    paddedAudio[start + i] * hannWindow[i]
                } else {
                    0f
                }
            }
            
            // Compute FFT for this frame
            val fftResult = rfft(frame)
            
            // Store result
            for (k in 0 until freqBins) {
                stft[frameIdx][k][0] = fftResult[k][0] // real
                stft[frameIdx][k][1] = fftResult[k][1] // imag
            }
        }
        
        return stft
    }
    
    /**
     * Real FFT using Cooley-Tukey algorithm.
     * Optimized for real-valued input.
     * 
     * @param x Real-valued input (length N_FFT)
     * @return Complex FFT output (length N_FFT/2 + 1, each with [real, imag])
     */
    private fun rfft(x: FloatArray): Array<FloatArray> {
        require(x.size == N_FFT) { "Input must be size $N_FFT" }
        
        // Convert to complex for standard FFT
        val complex = Array(N_FFT) { i -> floatArrayOf(x[i], 0f) }
        
        // Compute full FFT
        fft(complex, forward = true)
        
        // Return only first N_FFT/2 + 1 bins (rest are complex conjugates)
        return complex.sliceArray(0..(N_FFT / 2))
    }
    
    /**
     * Cooley-Tukey FFT algorithm (in-place).
     * 
     * @param x Complex array [real, imag]
     * @param forward true for forward FFT, false for inverse
     */
    private fun fft(x: Array<FloatArray>, forward: Boolean) {
        val n = x.size
        require(n and (n - 1) == 0) { "Size must be power of 2" }
        
        // Bit-reversal permutation
        var j = 0
        for (i in 0 until n - 1) {
            if (i < j) {
                val temp = x[i]
                x[i] = x[j]
                x[j] = temp
            }
            var k = n / 2
            while (k <= j) {
                j -= k
                k /= 2
            }
            j += k
        }
        
        // FFT butterfly
        val direction = if (forward) -1.0 else 1.0
        var len = 2
        while (len <= n) {
            val angle = 2.0 * PI * direction / len
            val wLen = floatArrayOf(cos(angle).toFloat(), sin(angle).toFloat())
            
            var i = 0
            while (i < n) {
                val w = floatArrayOf(1f, 0f)
                for (k in 0 until len / 2) {
                    val idx1 = i + k
                    val idx2 = i + k + len / 2
                    
                    // t = w * x[idx2]
                    val t = floatArrayOf(
                        w[0] * x[idx2][0] - w[1] * x[idx2][1],
                        w[0] * x[idx2][1] + w[1] * x[idx2][0]
                    )
                    
                    // x[idx2] = x[idx1] - t
                    x[idx2][0] = x[idx1][0] - t[0]
                    x[idx2][1] = x[idx1][1] - t[1]
                    
                    // x[idx1] = x[idx1] + t
                    x[idx1][0] += t[0]
                    x[idx1][1] += t[1]
                    
                    // w = w * wLen
                    val wTemp = floatArrayOf(
                        w[0] * wLen[0] - w[1] * wLen[1],
                        w[0] * wLen[1] + w[1] * wLen[0]
                    )
                    w[0] = wTemp[0]
                    w[1] = wTemp[1]
                }
                i += len
            }
            len *= 2
        }
        
        // Scale for inverse FFT
        if (!forward) {
            for (i in x.indices) {
                x[i][0] /= n
                x[i][1] /= n
            }
        }
    }
    
    /**
     * Convert complex STFT to power spectrogram.
     * 
     * @param stft Complex STFT [time][freq][2]
     * @return Power spectrogram [time][freq]
     */
    private fun stftToPower(stft: Array<Array<FloatArray>>): Array<FloatArray> {
        return Array(stft.size) { t ->
            FloatArray(stft[t].size) { f ->
                val real = stft[t][f][0]
                val imag = stft[t][f][1]
                real * real + imag * imag
            }
        }
    }
    
    /**
     * Create mel filterbank matrix.
     * 
     * @return Mel filterbank [n_mels][n_fft/2 + 1]
     */
    private fun createMelFilterbank(): Array<FloatArray> {
        val freqBins = N_FFT / 2 + 1
        
        // Mel scale conversion functions
        fun hzToMel(hz: Double) = 2595.0 * log10(1.0 + hz / 700.0)
        fun melToHz(mel: Double) = 700.0 * (10.0.pow(mel / 2595.0) - 1.0)
        
        // Create mel points
        val melMin = hzToMel(MEL_MIN_FREQ)
        val melMax = hzToMel(MEL_MAX_FREQ)
        val melPoints = FloatArray(N_MELS + 2) { i ->
            melToHz(melMin + i * (melMax - melMin) / (N_MELS + 1)).toFloat()
        }
        
        // Convert to FFT bin numbers
        val fftBins = melPoints.map { hz ->
            floor(N_FFT * hz / SAMPLE_RATE).toInt().coerceIn(0, N_FFT / 2)
        }
        
        // Create filterbank
        val filterbank = Array(N_MELS) { FloatArray(freqBins) }
        
        for (i in 0 until N_MELS) {
            val leftBin = fftBins[i]
            val centerBin = fftBins[i + 1]
            val rightBin = fftBins[i + 2]
            
            // Triangular filter
            for (j in leftBin until centerBin) {
                if (centerBin > leftBin) {
                    filterbank[i][j] = (j - leftBin).toFloat() / (centerBin - leftBin)
                }
            }
            for (j in centerBin until rightBin) {
                if (rightBin > centerBin) {
                    filterbank[i][j] = (rightBin - j).toFloat() / (rightBin - centerBin)
                }
            }
        }
        
        return filterbank
    }
    
    /**
     * Apply mel filterbank to power spectrogram.
     * 
     * @param powerSpec Power spectrogram [time][freq]
     * @return Mel spectrogram [time][n_mels]
     */
    private fun applyMelFilterbank(powerSpec: Array<FloatArray>): Array<FloatArray> {
        return Array(powerSpec.size) { t ->
            FloatArray(N_MELS) { m ->
                var sum = 0f
                for (f in melFilterbank[m].indices) {
                    sum += melFilterbank[m][f] * powerSpec[t][f]
                }
                sum
            }
        }
    }
}

