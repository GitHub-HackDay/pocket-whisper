package com.pocketwhisper.app.audio

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.isActive
import kotlin.coroutines.coroutineContext

/**
 * Captures raw audio from the microphone and emits 32ms chunks.
 * 
 * Audio format:
 * - Sample rate: 16000 Hz
 * - Channels: Mono
 * - Encoding: PCM 16-bit
 * - Chunk size: 512 samples (32ms)
 * 
 * Output is normalized to [-1.0, 1.0] floats for ML models.
 */
class AudioCaptureService {
    
    private val TAG = "AudioCaptureService"
    
    // Audio configuration
    private val SAMPLE_RATE = 16000
    private val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
    private val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    private val CHUNK_SIZE = 512 // 32ms at 16kHz
    
    // Calculate buffer size
    private val bufferSize = AudioRecord.getMinBufferSize(
        SAMPLE_RATE,
        CHANNEL_CONFIG,
        AUDIO_FORMAT
    ).coerceAtLeast(CHUNK_SIZE * 4) // At least 4 chunks
    
    private var audioRecord: AudioRecord? = null
    
    /**
     * Start capturing audio and emit normalized chunks.
     * 
     * @return Flow of audio chunks (512 samples each, normalized to [-1.0, 1.0])
     * @throws SecurityException if RECORD_AUDIO permission is not granted
     */
    fun captureAudio(): Flow<FloatArray> = flow {
        try {
            Log.d(TAG, "Starting audio capture...")
            Log.d(TAG, "Sample rate: $SAMPLE_RATE Hz")
            Log.d(TAG, "Buffer size: $bufferSize samples")
            Log.d(TAG, "Chunk size: $CHUNK_SIZE samples (32ms)")
            
            // Create AudioRecord instance
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION, // Optimized for speech
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize
            )
            
            val state = audioRecord?.state
            if (state != AudioRecord.STATE_INITIALIZED) {
                throw IllegalStateException("AudioRecord not initialized (state=$state)")
            }
            
            // Start recording
            audioRecord?.startRecording()
            Log.d(TAG, "AudioRecord started")
            
            // Read audio in chunks
            val shortBuffer = ShortArray(CHUNK_SIZE)
            val floatBuffer = FloatArray(CHUNK_SIZE)
            
            var chunksRead = 0
            
            while (coroutineContext.isActive) {
                // Read 16-bit PCM samples
                val samplesRead = audioRecord?.read(shortBuffer, 0, CHUNK_SIZE) ?: 0
                
                if (samplesRead < 0) {
                    Log.e(TAG, "AudioRecord.read() failed with error code: $samplesRead")
                    break
                }
                
                if (samplesRead != CHUNK_SIZE) {
                    Log.w(TAG, "Incomplete chunk: read $samplesRead samples, expected $CHUNK_SIZE")
                    continue
                }
                
                // Convert PCM16 to normalized floats [-1.0, 1.0]
                for (i in 0 until CHUNK_SIZE) {
                    floatBuffer[i] = shortBuffer[i] / 32768.0f
                }
                
                // Emit normalized chunk
                emit(floatBuffer.copyOf())
                
                chunksRead++
                if (chunksRead % 100 == 0) {
                    Log.d(TAG, "Captured $chunksRead chunks (${chunksRead * 32}ms)")
                }
            }
            
            Log.d(TAG, "Audio capture stopped. Total chunks: $chunksRead")
            
        } catch (e: SecurityException) {
            Log.e(TAG, "RECORD_AUDIO permission not granted", e)
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Audio capture failed", e)
            throw e
        } finally {
            stopCapture()
        }
    }.flowOn(Dispatchers.IO) // Run on IO dispatcher
    
    /**
     * Stop audio capture and release resources.
     */
    fun stopCapture() {
        Log.d(TAG, "Stopping audio capture...")
        audioRecord?.apply {
            if (state == AudioRecord.STATE_INITIALIZED) {
                try {
                    stop()
                    Log.d(TAG, "AudioRecord stopped")
                } catch (e: IllegalStateException) {
                    Log.w(TAG, "AudioRecord.stop() failed", e)
                }
            }
            release()
            Log.d(TAG, "AudioRecord released")
        }
        audioRecord = null
    }
    
    /**
     * Check if audio recording is supported on this device.
     * 
     * @return true if the required sample rate and format are supported
     */
    fun isSupported(): Boolean {
        val minBufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT
        )
        return minBufferSize != AudioRecord.ERROR_BAD_VALUE &&
               minBufferSize != AudioRecord.ERROR
    }
    
    /**
     * Get the current recording state.
     * 
     * @return true if actively recording
     */
    fun isRecording(): Boolean {
        return audioRecord?.recordingState == AudioRecord.RECORDSTATE_RECORDING
    }
}

