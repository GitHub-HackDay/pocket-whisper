package com.pocketwhisper.app.audio

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max

class AudioCaptureService {
    
    private val TAG = "AudioCapture"
    
    // Audio config: 16kHz mono PCM (standard for speech models)
    private val SAMPLE_RATE = 16000
    private val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
    private val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    
    // Buffer size: 512 samples (32ms at 16kHz) for VAD
    private val CHUNK_SIZE = 512
    private val BUFFER_SIZE_IN_BYTES = max(
        AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT),
        CHUNK_SIZE * 2  // 2 bytes per sample for 16-bit PCM
    )
    
    private var audioRecord: AudioRecord? = null
    private var captureJob: Job? = null
    
    // Flow of audio chunks (as FloatArray normalized to [-1, 1])
    private val _audioFlow = MutableSharedFlow<FloatArray>(replay = 0)
    val audioFlow: SharedFlow<FloatArray> = _audioFlow
    
    // Status flow
    private val _statusFlow = MutableSharedFlow<String>(replay = 1)
    val statusFlow: SharedFlow<String> = _statusFlow
    
    /**
     * Check if we have microphone permission and can record
     */
    fun canRecord(): Boolean {
        return try {
            val testRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                BUFFER_SIZE_IN_BYTES
            )
            val canInit = testRecord.state == AudioRecord.STATE_INITIALIZED
            testRecord.release()
            canInit
        } catch (e: Exception) {
            Log.e(TAG, "Cannot initialize AudioRecord: ${e.message}")
            false
        }
    }
    
    /**
     * Start capturing audio from microphone.
     */
    fun startCapture(scope: CoroutineScope) {
        if (!canRecord()) {
            Log.e(TAG, "Cannot record audio - check microphone permission")
            scope.launch {
                _statusFlow.emit("❌ Microphone permission denied")
            }
            return
        }
        
        Log.d(TAG, "Starting audio capture...")
        Log.d(TAG, "Sample rate: $SAMPLE_RATE Hz")
        Log.d(TAG, "Chunk size: $CHUNK_SIZE samples (${CHUNK_SIZE * 1000 / SAMPLE_RATE}ms)")
        Log.d(TAG, "Buffer size: $BUFFER_SIZE_IN_BYTES bytes")
        
        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                BUFFER_SIZE_IN_BYTES
            )
            
            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord initialization failed")
                scope.launch {
                    _statusFlow.emit("❌ Failed to initialize microphone")
                }
                return
            }
            
            audioRecord?.startRecording()
            scope.launch {
                _statusFlow.emit("🎤 Recording from microphone...")
            }
            
            captureJob = scope.launch(Dispatchers.IO) {
                val buffer = ShortArray(CHUNK_SIZE)
                var chunkCount = 0
                
                while (isActive) {
                    val readResult = audioRecord?.read(buffer, 0, CHUNK_SIZE) ?: 0
                    
                    when {
                        readResult > 0 -> {
                            // Convert 16-bit PCM to normalized float
                            val floatBuffer = FloatArray(readResult) { i ->
                                buffer[i].toFloat() / 32768.0f  // Normalize to [-1, 1]
                            }
                            
                            // Calculate audio statistics
                            val maxAmplitude = floatBuffer.maxOfOrNull { kotlin.math.abs(it) } ?: 0f
                            val avgAmplitude = floatBuffer.map { kotlin.math.abs(it) }.average().toFloat()
                            val rms = kotlin.math.sqrt(floatBuffer.map { it * it }.average()).toFloat()
                            
                            // Log every 10th chunk with detailed stats
                            if (++chunkCount % 10 == 0) {
                                Log.d(TAG, "Audio chunk #$chunkCount: max=${(maxAmplitude * 100).toInt()}%, " +
                                          "avg=${(avgAmplitude * 100).toInt()}%, rms=${(rms * 100).toInt()}%")
                                
                                // Check if audio seems too quiet
                                if (maxAmplitude < 0.01f) {
                                    Log.w(TAG, "Audio seems very quiet! Check microphone permission and volume.")
                                }
                            }
                            
                            // Emit the chunk for processing
                            _audioFlow.emit(floatBuffer)
                        }
                        readResult == AudioRecord.ERROR_INVALID_OPERATION -> {
                            Log.e(TAG, "AudioRecord ERROR_INVALID_OPERATION")
                            _statusFlow.emit("❌ Audio recording error")
                            break
                        }
                        readResult == AudioRecord.ERROR_BAD_VALUE -> {
                            Log.e(TAG, "AudioRecord ERROR_BAD_VALUE")
                            _statusFlow.emit("❌ Invalid audio parameters")
                            break
                        }
                    }
                }
            }
            
            Log.d(TAG, "✅ Audio capture started")
        } catch (e: SecurityException) {
            Log.e(TAG, "Security exception: ${e.message}")
            scope.launch {
                _statusFlow.emit("❌ Microphone permission required")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start audio capture: ${e.message}", e)
            scope.launch {
                _statusFlow.emit("❌ Error: ${e.message}")
            }
        }
    }
    
    /**
     * Stop capturing audio.
     */
    fun stopCapture() {
        Log.d(TAG, "Stopping audio capture...")
        captureJob?.cancel()
        captureJob = null
        
        audioRecord?.apply {
            try {
                if (recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                    stop()
                }
                release()
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping audio record: ${e.message}")
            }
        }
        audioRecord = null
        
        GlobalScope.launch {
            _statusFlow.emit("⏹️ Recording stopped")
        }
        
        Log.d(TAG, "✅ Audio capture stopped")
    }
    
    /**
     * Check if currently recording
     */
    fun isRecording(): Boolean {
        return audioRecord?.recordingState == AudioRecord.RECORDSTATE_RECORDING
    }
}
