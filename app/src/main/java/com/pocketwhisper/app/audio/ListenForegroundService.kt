package com.pocketwhisper.app.audio

import android.app.*
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import com.pocketwhisper.app.MainActivity
import com.pocketwhisper.app.ml.VadDetector
import com.pocketwhisper.app.ml.AsrSession
import com.pocketwhisper.app.ml.LlmSession
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.catch
import java.util.LinkedList

/**
 * Foreground service that orchestrates the entire speech-to-suggestion pipeline.
 * 
 * Pipeline:
 * 1. AudioCaptureService → raw audio chunks (32ms)
 * 2. VadDetector → detect speech vs silence
 * 3. Accumulate speech chunks → audio buffer
 * 4. AsrSession → transcribe speech to text
 * 5. LlmSession → predict next word
 * 6. Emit suggestion events
 * 
 * Runs as a foreground service with a persistent notification.
 */
class ListenForegroundService : Service() {
    
    private val TAG = "ListenForegroundService"
    private val NOTIFICATION_ID = 1001
    private val CHANNEL_ID = "pocket_whisper_listen"
    
    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private var audioJob: Job? = null
    
    // Components
    private lateinit var audioCaptureService: AudioCaptureService
    private lateinit var vadDetector: VadDetector
    private lateinit var asrSession: AsrSession
    private lateinit var llmSession: LlmSession
    
    // Speech buffer
    private val speechBuffer = LinkedList<FloatArray>()
    private var isSpeechActive = false
    private var silenceChunks = 0
    private val MAX_SILENCE_CHUNKS = 10 // ~320ms of silence ends speech
    
    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "Service created")
        
        try {
            // Initialize all ML components
            audioCaptureService = AudioCaptureService()
            vadDetector = VadDetector(this)
            asrSession = AsrSession(this)
            llmSession = LlmSession(this)
            
            createNotificationChannel()
            Log.d(TAG, "All components initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize components", e)
            stopSelf()
        }
    }
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "Service started")
        
        // Start as foreground service
        val notification = createNotification()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
        
        // Start audio processing pipeline
        startListening()
        
        return START_STICKY
    }
    
    /**
     * Start the audio processing pipeline.
     */
    private fun startListening() {
        Log.d(TAG, "Starting audio pipeline...")
        
        audioJob = serviceScope.launch {
            audioCaptureService.captureAudio()
                .catch { e ->
                    Log.e(TAG, "Audio capture error", e)
                }
                .collect { audioChunk ->
                    processAudioChunk(audioChunk)
                }
        }
    }
    
    /**
     * Process a single audio chunk through the pipeline.
     */
    private suspend fun processAudioChunk(audioChunk: FloatArray) {
        try {
            // 1. Run VAD to detect speech
            val speechProb = vadDetector.detectSpeech(audioChunk)
            val isSpeech = speechProb > 0.5f
            
            if (isSpeech) {
                // Speech detected
                if (!isSpeechActive) {
                    Log.d(TAG, "🎤 Speech started")
                    isSpeechActive = true
                    speechBuffer.clear()
                }
                
                // Add to buffer
                speechBuffer.add(audioChunk)
                silenceChunks = 0
                
            } else {
                // Silence detected
                if (isSpeechActive) {
                    silenceChunks++
                    
                    // Add silence chunks (for natural speech end)
                    if (silenceChunks <= MAX_SILENCE_CHUNKS) {
                        speechBuffer.add(audioChunk)
                    }
                    
                    // End of speech detected
                    if (silenceChunks >= MAX_SILENCE_CHUNKS) {
                        Log.d(TAG, "🛑 Speech ended (${speechBuffer.size} chunks)")
                        processSpeechSegment()
                        
                        isSpeechActive = false
                        speechBuffer.clear()
                        silenceChunks = 0
                    }
                }
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error processing audio chunk", e)
        }
    }
    
    /**
     * Process complete speech segment: ASR + LLM.
     */
    private suspend fun processSpeechSegment() {
        if (speechBuffer.isEmpty()) {
            Log.w(TAG, "Empty speech buffer")
            return
        }
        
        try {
            Log.d(TAG, "Processing speech segment...")
            
            // 1. Flatten buffer to single audio array
            val totalSamples = speechBuffer.sumOf { it.size }
            val audioArray = FloatArray(totalSamples)
            var offset = 0
            for (chunk in speechBuffer) {
                chunk.copyInto(audioArray, offset)
                offset += chunk.size
            }
            
            val durationMs = (totalSamples / 16.0).toInt() // 16kHz
            Log.d(TAG, "Audio segment: ${totalSamples} samples (${durationMs}ms)")
            
            // 2. Transcribe with ASR
            Log.d(TAG, "📝 Transcribing...")
            val transcription = asrSession.transcribe(audioArray)
            Log.d(TAG, "Transcription: '$transcription'")
            
            if (transcription.isBlank()) {
                Log.d(TAG, "Empty transcription, skipping LLM")
                return
            }
            
            // 3. Predict next word with LLM
            Log.d(TAG, "🤖 Predicting next word...")
            val nextWord = llmSession.predictNextWord(transcription)
            val confidence = llmSession.getConfidence(transcription)
            
            Log.d(TAG, "💡 Suggestion: '$nextWord' (confidence: ${(confidence * 100).toInt()}%)")
            
            // TODO: Emit suggestion event to UI
            // TODO: Send to Accessibility Service for auto-insert
            // TODO: Trigger TTS if enabled
            
        } catch (e: Exception) {
            Log.e(TAG, "Error processing speech segment", e)
        }
    }
    
    override fun onDestroy() {
        Log.d(TAG, "Service destroyed")
        
        // Cancel audio processing
        audioJob?.cancel()
        audioCaptureService.stopCapture()
        
        // Clean up ML components
        vadDetector.close()
        asrSession.close()
        llmSession.close()
        
        serviceScope.cancel()
        super.onDestroy()
    }
    
    override fun onBind(intent: Intent?): IBinder? = null
    
    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Pocket Whisper Listening",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Shows when Pocket Whisper is listening"
        }
        
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(channel)
    }
    
    private fun createNotification(): Notification {
        val intent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent, PendingIntent.FLAG_IMMUTABLE
        )
        
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Pocket Whisper")
            .setContentText("Listening for speech...")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }
}
