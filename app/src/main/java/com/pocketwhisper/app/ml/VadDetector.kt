package com.pocketwhisper.app.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Voice Activity Detection using Silero VAD (ONNX model).
 * 
 * Detects speech vs silence in 32ms audio chunks (512 samples at 16kHz).
 * Uses LSTM state to maintain context across chunks.
 * 
 * Model: https://github.com/snakers4/silero-vad
 */
class VadDetector(context: Context) {
    
    private val TAG = "VadDetector"
    
    private val env = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    
    // LSTM state (2, 1, 128) - maintains context across chunks
    private var state: FloatArray = FloatArray(2 * 1 * 128) { 0f }
    
    // Model constants
    private val SAMPLE_RATE = 16000L
    private val CHUNK_SIZE = 512 // 32ms at 16kHz
    
    // Detection threshold
    private val SPEECH_THRESHOLD = 0.5f
    
    init {
        loadModel(context)
    }
    
    private fun loadModel(context: Context) {
        try {
            Log.d(TAG, "Loading Silero VAD model...")
            val startTime = System.currentTimeMillis()
            
            val modelLoader = ModelLoader(context)
            val modelPath = modelLoader.loadModel("vad_silero.onnx")
            
            // Create ONNX session
            val sessionOptions = OrtSession.SessionOptions()
            session = env.createSession(modelPath, sessionOptions)
            
            val elapsed = System.currentTimeMillis() - startTime
            Log.d(TAG, "VAD model loaded in ${elapsed}ms")
            
            // Log model info
            Log.d(TAG, "Input names: ${session?.inputNames}")
            Log.d(TAG, "Output names: ${session?.outputNames}")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load VAD model", e)
            throw e
        }
    }
    
    /**
     * Detect speech in an audio chunk.
     * 
     * @param audioChunk 512 samples (32ms) of normalized audio [-1.0, 1.0]
     * @return Speech probability [0.0, 1.0]
     * @throws IllegalArgumentException if chunk size is incorrect
     */
    suspend fun detectSpeech(audioChunk: FloatArray): Float = withContext(Dispatchers.Default) {
        require(audioChunk.size == CHUNK_SIZE) {
            "Expected $CHUNK_SIZE samples, got ${audioChunk.size}"
        }
        
        try {
            val currentSession = session ?: throw IllegalStateException("Model not loaded")
            
            // Prepare inputs
            // input: (1, 512) - audio chunk
            val inputBuffer = FloatBuffer.wrap(audioChunk)
            val inputTensor = OnnxTensor.createTensor(
                env,
                inputBuffer,
                longArrayOf(1, CHUNK_SIZE.toLong())
            )
            
            // sr: (1,) - sample rate
            val srBuffer = LongBuffer.wrap(longArrayOf(SAMPLE_RATE))
            val srTensor = OnnxTensor.createTensor(
                env,
                srBuffer,
                longArrayOf(1)
            )
            
            // state: (2, 1, 128) - LSTM state from previous chunk
            val stateBuffer = FloatBuffer.wrap(state)
            val stateTensor = OnnxTensor.createTensor(
                env,
                stateBuffer,
                longArrayOf(2, 1, 128)
            )
            
            // Run inference
            val inputs = mapOf(
                "input" to inputTensor,
                "sr" to srTensor,
                "state" to stateTensor
            )
            
            val outputs = currentSession.run(inputs)
            
            // Get outputs
            // output: (1,) - speech probability
            val outputTensor = outputs[0] as OnnxTensor
            val speechProb = outputTensor.floatBuffer.get(0)
            
            // new_state: (2, 1, 128) - updated LSTM state
            val newStateTensor = outputs[1] as OnnxTensor
            val newStateBuffer = newStateTensor.floatBuffer
            newStateBuffer.get(state) // Update state for next chunk
            
            // Clean up tensors
            inputTensor.close()
            srTensor.close()
            stateTensor.close()
            outputs.forEach { it.value.close() }
            
            speechProb
            
        } catch (e: Exception) {
            Log.e(TAG, "VAD inference failed", e)
            throw e
        }
    }
    
    /**
     * Check if the chunk contains speech.
     * 
     * @param audioChunk 512 samples of normalized audio
     * @return true if speech is detected (prob > threshold)
     */
    suspend fun isSpeech(audioChunk: FloatArray): Boolean {
        val prob = detectSpeech(audioChunk)
        return prob > SPEECH_THRESHOLD
    }
    
    /**
     * Reset the LSTM state.
     * Call this when starting a new audio stream or after long silence.
     */
    fun resetState() {
        Log.d(TAG, "Resetting VAD state")
        state.fill(0f)
    }
    
    /**
     * Clean up resources.
     */
    fun close() {
        Log.d(TAG, "Closing VAD detector")
        session?.close()
        session = null
    }
}

