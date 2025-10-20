package com.pocketwhisper.app.ml

import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor

class VadDetector(private val module: Module) {
    
    private val TAG = "VadDetector"
    
    // VAD model expects 512 samples (32ms at 16kHz)
    private val CHUNK_SIZE = 512
    
    init {
        Log.d(TAG, "VadDetector initialized with module: $module")
        try {
            Log.d(TAG, "Module type: ${module.javaClass.name}")
        } catch (e: Exception) {
            Log.e(TAG, "Error getting module info: ${e.message}", e)
        }
    }
    
    /**
     * Run VAD on audio chunk.
     * Returns probability of speech (0.0 = silence, 1.0 = speech).
     */
    fun detectSpeech(audioChunk: FloatArray): Float {
        Log.d(TAG, "detectSpeech called with audio chunk size: ${audioChunk.size}")
        
        return try {
            // Ensure chunk is 512 samples (32ms at 16kHz)
            val paddedChunk = when {
                audioChunk.size < CHUNK_SIZE -> {
                    Log.d(TAG, "Padding audio chunk from ${audioChunk.size} to $CHUNK_SIZE")
                    FloatArray(CHUNK_SIZE).apply {
                        audioChunk.copyInto(this)
                    }
                }
                audioChunk.size > CHUNK_SIZE -> {
                    Log.d(TAG, "Truncating audio chunk from ${audioChunk.size} to $CHUNK_SIZE")
                    audioChunk.sliceArray(0 until CHUNK_SIZE)
                }
                else -> {
                    Log.d(TAG, "Audio chunk already correct size: $CHUNK_SIZE")
                    audioChunk
                }
            }
            
            Log.d(TAG, "Creating input tensor with shape [1, $CHUNK_SIZE]")
            // Create input tensor: shape (1, 512)
            val inputTensor = try {
                Tensor.fromBlob(
                    paddedChunk,
                    longArrayOf(1, CHUNK_SIZE.toLong())
                )
            } catch (e: Exception) {
                Log.e(TAG, "Failed to create tensor: ${e.message}", e)
                throw e
            }
            
            Log.d(TAG, "Input tensor created successfully")
            Log.d(TAG, "Tensor shape: ${inputTensor.shape().contentToString()}")
            Log.d(TAG, "Tensor dtype: ${inputTensor.dtype()}")
            
            // Run inference - Silero VAD requires sample rate as second argument
            Log.d(TAG, "Running model forward pass with sample rate...")
            val output = try {
                // Create IValues for both arguments
                val tensorIValue = IValue.from(inputTensor)
                val sampleRateIValue = IValue.from(16000L)  // 16kHz sample rate
                
                Log.d(TAG, "IValue tensor created: $tensorIValue")
                Log.d(TAG, "IValue sample rate: 16000")
                
                // Pass both arguments to the model
                module.forward(tensorIValue, sampleRateIValue)
            } catch (e: Exception) {
                Log.e(TAG, "Model forward pass failed: ${e.message}", e)
                Log.e(TAG, "Stack trace:", e)
                throw e
            }
            
            Log.d(TAG, "Model forward pass completed")
            Log.d(TAG, "Output type: ${output.javaClass.name}")
            
            // Extract speech probability
            val speechProb = try {
                val outputTensor = output.toTensor()
                Log.d(TAG, "Output tensor shape: ${outputTensor.shape().contentToString()}")
                Log.d(TAG, "Output tensor dtype: ${outputTensor.dtype()}")
                
                val dataArray = outputTensor.dataAsFloatArray
                Log.d(TAG, "Output data array size: ${dataArray.size}")
                Log.d(TAG, "Output values: ${dataArray.take(10).joinToString()}")
                
                if (dataArray.isEmpty()) {
                    Log.e(TAG, "Output tensor is empty!")
                    0f
                } else {
                    dataArray[0]
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to extract result from output: ${e.message}", e)
                throw e
            }
            
            Log.d(TAG, "Speech probability: %.3f".format(speechProb))
            speechProb
            
        } catch (e: Exception) {
            Log.e(TAG, "ERROR in detectSpeech: ${e.message}", e)
            Log.e(TAG, "Full exception: ", e)
            // Return a default value instead of crashing
            0.5f
        }
    }
    
    /**
     * Process multiple chunks and return average probability
     */
    fun detectSpeechBatch(audioChunks: List<FloatArray>): Float {
        Log.d(TAG, "detectSpeechBatch called with ${audioChunks.size} chunks")
        return try {
            val probs = audioChunks.map { detectSpeech(it) }
            probs.average().toFloat()
        } catch (e: Exception) {
            Log.e(TAG, "Error in detectSpeechBatch: ${e.message}", e)
            0.5f
        }
    }
}