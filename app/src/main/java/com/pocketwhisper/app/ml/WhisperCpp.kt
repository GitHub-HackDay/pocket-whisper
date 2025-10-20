package com.pocketwhisper.app.ml

import android.content.Context
import android.util.Log
import java.io.File

/**
 * JNI wrapper for whisper.cpp native library.
 * 
 * Provides Kotlin bindings to the high-performance C++ implementation of OpenAI Whisper.
 * 
 * Benefits over PyTorch:
 * - 3-5x faster inference
 * - Lower memory usage
 * - Production-proven (used by thousands of apps)
 * - Full encoder-decoder architecture
 * - No export complexity
 * 
 * Model: Distil-Whisper Small (~75MB in ggml format)
 */
class WhisperCpp(context: Context) {
    
    private val TAG = "WhisperCpp"
    
    // Native context pointer
    private var contextPtr: Long = 0
    
    // Load native library
    init {
        try {
            System.loadLibrary("whisper")
            Log.d(TAG, "✓ Native library loaded")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Failed to load whisper native library", e)
            throw RuntimeException("whisper.cpp native library not found. Did you run setup_whisper_cpp.sh?", e)
        }
        
        // Load model
        val modelPath = loadModel(context)
        contextPtr = nativeInit(modelPath)
        
        if (contextPtr == 0L) {
            throw RuntimeException("Failed to initialize whisper.cpp context")
        }
        
        Log.d(TAG, "✓ Whisper context initialized")
    }
    
    /**
     * Copy model from assets to files directory (native code needs file path).
     */
    private fun loadModel(context: Context): String {
        val modelName = "ggml-distil-small.en.bin"
        val modelFile = File(context.filesDir, modelName)
        
        if (!modelFile.exists()) {
            Log.d(TAG, "Copying model from assets...")
            val startTime = System.currentTimeMillis()
            
            context.assets.open(modelName).use { input ->
                modelFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            
            val elapsed = System.currentTimeMillis() - startTime
            val sizeMB = modelFile.length() / (1024.0 * 1024.0)
            Log.d(TAG, "✓ Model copied: ${sizeMB.format(1)} MB in ${elapsed}ms")
        }
        
        return modelFile.absolutePath
    }
    
    /**
     * Transcribe audio to text.
     * 
     * @param audioData 16kHz mono PCM audio as floats [-1.0, 1.0]
     * @return Transcribed text
     */
    fun transcribe(audioData: FloatArray): String {
        if (contextPtr == 0L) {
            throw IllegalStateException("Whisper context not initialized")
        }
        
        val startTime = System.currentTimeMillis()
        val result = nativeTranscribe(contextPtr, audioData)
        val elapsed = System.currentTimeMillis() - startTime
        
        Log.d(TAG, "Transcription: '$result' (${elapsed}ms)")
        
        return result
    }
    
    /**
     * Transcribe with custom parameters.
     * 
     * @param audioData Audio samples
     * @param language Language code (default: "en")
     * @param translate Translate to English (default: false)
     * @param noContext Disable context from previous transcription (default: false)
     * @param singleSegment Force single segment output (default: false)
     * @return Transcribed text
     */
    fun transcribe(
        audioData: FloatArray,
        language: String = "en",
        translate: Boolean = false,
        noContext: Boolean = false,
        singleSegment: Boolean = false
    ): String {
        if (contextPtr == 0L) {
            throw IllegalStateException("Whisper context not initialized")
        }
        
        val params = WhisperParams(
            language = language,
            translate = translate,
            noContext = noContext,
            singleSegment = singleSegment
        )
        
        val startTime = System.currentTimeMillis()
        val result = nativeTranscribeWithParams(contextPtr, audioData, params)
        val elapsed = System.currentTimeMillis() - startTime
        
        Log.d(TAG, "Transcription: '$result' (${elapsed}ms)")
        
        return result
    }
    
    /**
     * Get transcription with timing information for each segment.
     * 
     * @param audioData Audio samples
     * @return List of segments with text and timestamps
     */
    fun transcribeWithTimestamps(audioData: FloatArray): List<WhisperSegment> {
        if (contextPtr == 0L) {
            throw IllegalStateException("Whisper context not initialized")
        }
        
        return nativeTranscribeWithTimestamps(contextPtr, audioData)
    }
    
    /**
     * Clean up native resources.
     * IMPORTANT: Call this when done to avoid memory leaks!
     */
    fun close() {
        if (contextPtr != 0L) {
            nativeDestroy(contextPtr)
            contextPtr = 0
            Log.d(TAG, "✓ Whisper context destroyed")
        }
    }
    
    // Native method declarations
    private external fun nativeInit(modelPath: String): Long
    private external fun nativeTranscribe(contextPtr: Long, audioData: FloatArray): String
    private external fun nativeTranscribeWithParams(
        contextPtr: Long,
        audioData: FloatArray,
        params: WhisperParams
    ): String
    private external fun nativeTranscribeWithTimestamps(
        contextPtr: Long,
        audioData: FloatArray
    ): List<WhisperSegment>
    private external fun nativeDestroy(contextPtr: Long)
    
    private fun Double.format(decimals: Int) = "%.${decimals}f".format(this)
    
    // Ensure cleanup on garbage collection
    protected fun finalize() {
        close()
    }
}

/**
 * Whisper transcription parameters.
 */
data class WhisperParams(
    val language: String = "en",
    val translate: Boolean = false,
    val noContext: Boolean = false,
    val singleSegment: Boolean = false,
    val maxTokens: Int = 0,  // 0 = no limit
    val beamSize: Int = 5,
    val audioCtx: Int = 0  // 0 = use default
)

/**
 * Transcription segment with timing information.
 */
data class WhisperSegment(
    val text: String,
    val startMs: Long,
    val endMs: Long,
    val speakerTurn: Boolean = false
)

/**
 * Exception thrown when whisper.cpp encounters an error.
 */
class WhisperException(message: String, cause: Throwable? = null) : Exception(message, cause)

