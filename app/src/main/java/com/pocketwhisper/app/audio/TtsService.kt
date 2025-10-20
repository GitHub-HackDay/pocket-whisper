package com.pocketwhisper.app.audio

import android.content.Context
import android.media.AudioManager
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import kotlinx.coroutines.suspendCancellableCoroutine
import java.util.*
import kotlin.coroutines.resume

/**
 * Text-to-Speech service for speaking next-word suggestions.
 * Outputs through phone speaker (or earpiece if configured).
 */
class TtsService(private val context: Context) {
    
    private val TAG = "TtsService"
    private var tts: TextToSpeech? = null
    private var isReady = false
    private val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    
    // TTS configuration
    private var outputMode = OutputMode.SPEAKER
    
    enum class OutputMode {
        SPEAKER,     // Normal speaker output
        EARPIECE,    // Earpiece for privacy
        SILENT       // No audio (for testing)
    }
    
    suspend fun initialize(): Boolean = suspendCancellableCoroutine { continuation ->
        Log.d(TAG, "Initializing TTS...")
        
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.let { engine ->
                    // Configure TTS
                    engine.language = Locale.US
                    engine.setPitch(1.0f)
                    engine.setSpeechRate(1.1f) // Slightly faster for quick suggestions
                    
                    // Add listener for progress
                    engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                        override fun onStart(utteranceId: String?) {
                            Log.d(TAG, "TTS started: $utteranceId")
                        }
                        
                        override fun onDone(utteranceId: String?) {
                            Log.d(TAG, "TTS done: $utteranceId")
                        }
                        
                        override fun onError(utteranceId: String?) {
                            Log.e(TAG, "TTS error: $utteranceId")
                        }
                    })
                    
                    isReady = true
                    Log.d(TAG, "✅ TTS initialized successfully")
                    continuation.resume(true)
                }
            } else {
                Log.e(TAG, "❌ TTS initialization failed with status: $status")
                continuation.resume(false)
            }
        }
    }
    
    /**
     * Speak a word or short phrase through the speaker.
     * This is the main method for suggesting next words.
     */
    fun speak(text: String, utteranceId: String = "suggestion_${System.currentTimeMillis()}") {
        if (!isReady) {
            Log.w(TAG, "TTS not ready yet, cannot speak: $text")
            return
        }
        
        if (outputMode == OutputMode.SILENT) {
            Log.d(TAG, "Silent mode, skipping TTS: $text")
            return
        }
        
        // Configure audio routing based on output mode
        when (outputMode) {
            OutputMode.EARPIECE -> {
                // Route to earpiece for privacy
                audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
                audioManager.isSpeakerphoneOn = false
            }
            OutputMode.SPEAKER -> {
                // Route to speaker (default)
                audioManager.mode = AudioManager.MODE_NORMAL
                audioManager.isSpeakerphoneOn = true
            }
            OutputMode.SILENT -> {
                // No-op, already handled above
            }
        }
        
        val params = android.os.Bundle().apply {
            putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, utteranceId)
            putString(TextToSpeech.Engine.KEY_PARAM_VOLUME, "0.7") // Moderate volume
        }
        
        Log.d(TAG, "🔊 Speaking: '$text' (mode: $outputMode)")
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, params, utteranceId)
    }
    
    /**
     * Set output mode (speaker, earpiece, or silent)
     */
    fun setOutputMode(mode: OutputMode) {
        outputMode = mode
        Log.d(TAG, "Output mode set to: $mode")
    }
    
    /**
     * Stop any ongoing speech
     */
    fun stop() {
        tts?.stop()
    }
    
    /**
     * Cleanup resources
     */
    fun shutdown() {
        Log.d(TAG, "Shutting down TTS...")
        tts?.stop()
        tts?.shutdown()
        tts = null
        isReady = false
        
        // Reset audio mode
        audioManager.mode = AudioManager.MODE_NORMAL
    }
    
    fun isInitialized(): Boolean = isReady
}

