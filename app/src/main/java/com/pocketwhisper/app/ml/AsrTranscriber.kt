package com.pocketwhisper.app.ml

import android.content.Context
import android.util.Log
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream

class AsrTranscriber(private val context: Context) {
    
    private val TAG = "AsrTranscriber"
    private var module: Module? = null
    private val bufferManager = AsrBufferManager()
    private var lastTranscriptionTime = 0L
    private val decoder = Wav2Vec2Decoder()
    
    init {
        loadModel()
    }
    
    private fun loadModel() {
        try {
            Log.d(TAG, "Loading ASR model...")
            
            // Check if model exists in assets
            val assetList = context.assets.list("") ?: emptyArray()
            val hasAsrModel = assetList.contains("asr_wav2vec2_base.pt")
            
            if (!hasAsrModel) {
                Log.e(TAG, "ASR model not found in assets!")
                return
            }
            
            // Copy model from assets to cache in chunks (for large files)
            val modelPath = File(context.filesDir, "asr_wav2vec2_base.pt")
            
            if (!modelPath.exists()) {
                Log.d(TAG, "Copying ASR model to cache (this may take a minute)...")
                try {
                    context.assets.open("asr_wav2vec2_base.pt").use { input ->
                        FileOutputStream(modelPath).use { output ->
                            val buffer = ByteArray(8192)  // 8KB buffer
                            var bytesRead: Int
                            var totalBytes = 0L
                            
                            while (input.read(buffer).also { bytesRead = it } != -1) {
                                output.write(buffer, 0, bytesRead)
                                totalBytes += bytesRead
                                
                                // Log progress every 10MB
                                if (totalBytes % (10 * 1024 * 1024) == 0L) {
                                    Log.d(TAG, "Copied ${totalBytes / 1024 / 1024} MB...")
                                }
                            }
                        }
                    }
                    Log.d(TAG, "Model copied, size: ${modelPath.length() / 1024 / 1024} MB")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to copy model", e)
                    modelPath.delete()  // Clean up partial file
                    return
                }
            } else {
                Log.d(TAG, "Using cached model (${modelPath.length() / 1024 / 1024} MB)")
            }
            
            // Load the model using PyTorch Lite
            Log.d(TAG, "Loading model from: ${modelPath.absolutePath}")
            try {
                module = LiteModuleLoader.load(modelPath.absolutePath)
                Log.d(TAG, "✅ ASR model loaded successfully!")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load model file", e)
                // Try deleting and re-copying next time
                modelPath.delete()
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load ASR model", e)
        }
    }
    
    /**
     * Add audio chunk and transcribe when buffer is ready
     * @param audioChunk Float array of audio samples (16kHz)
     * @return Transcribed text when buffer is full, null otherwise
     */
    fun transcribe(audioChunk: FloatArray): String? {
        // Add chunk to buffer
        val fullAudio = bufferManager.addChunk(audioChunk) ?: return null
        
        // Throttle transcription to avoid overload
        val now = System.currentTimeMillis()
        if (now - lastTranscriptionTime < 500) { // Max 2 transcriptions per second
            return null
        }
        lastTranscriptionTime = now
        
        return try {
            val model = module ?: run {
                Log.e(TAG, "Model not loaded")
                return null
            }
            
            Log.d(TAG, "Transcribing ${fullAudio.size} samples (${fullAudio.size/16000.0f}s)...")
            
            // Create input tensor
            val inputTensor = Tensor.fromBlob(
                fullAudio,
                longArrayOf(1, fullAudio.size.toLong())
            )
            
            // Run inference
            val startTime = System.currentTimeMillis()
            val output = model.forward(IValue.from(inputTensor))
            val inferenceTime = System.currentTimeMillis() - startTime
            
            Log.d(TAG, "Inference completed in ${inferenceTime}ms")
            
            // Get logits
            val outputTensor = output.toTensor()
            val logits = outputTensor.dataAsFloatArray
            val shape = outputTensor.shape()
            
            Log.d(TAG, "Output shape: ${shape.contentToString()}, size: ${logits.size}")
            
            // Decode logits to text
            val decodedText = try {
                decoder.quickDecode(logits)
            } catch (e: Exception) {
                Log.e(TAG, "Decoding failed", e)
                "[decode error]"
            }
            
            val result = if (decodedText.isNotEmpty()) {
                decodedText
            } else {
                "(silence)"
            }
            
            Log.d(TAG, "Transcription: '$result' (${inferenceTime}ms)")
            result
            
        } catch (e: Exception) {
            Log.e(TAG, "Transcription failed", e)
            null
        }
    }
    
    /**
     * Test the ASR model with dummy input
     */
    fun testModel(): Boolean {
        return try {
            Log.d(TAG, "Testing ASR model...")
            
            // Create dummy audio (1 second at 16kHz)
            val dummyAudio = FloatArray(16000) { 
                (Math.random() * 0.01).toFloat() - 0.005f 
            }
            
            val result = transcribe(dummyAudio)
            if (result != null) {
                Log.d(TAG, "✅ ASR test passed: $result")
                true
            } else {
                Log.e(TAG, "❌ ASR test failed: no result")
                false
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ ASR test failed", e)
            false
        }
    }
}
