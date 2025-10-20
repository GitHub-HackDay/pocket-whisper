package com.pocketwhisper.app.ml

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor

/**
 * LLM session using Qwen2-0.5B for next-word prediction.
 * 
 * Model: Qwen/Qwen2-0.5B-Instruct (TorchScript .ptl)
 * Runtime: PyTorch Mobile
 * 
 * Usage:
 * ```
 * val llm = LlmSession(context)
 * val nextWord = llm.predictNextWord("Could you please")
 * // Returns: " provide" (or similar)
 * ```
 */
class LlmSession(context: Context) {
    
    private val TAG = "LlmSession"
    
    // Components
    private val tokenizer = QwenTokenizer(context)
    private var model: Module? = null
    
    // Model constants
    private val VOCAB_SIZE = 151936 // Qwen2 vocabulary size
    private val MAX_LENGTH = 512 // Maximum sequence length
    
    init {
        loadModel(context)
    }
    
    /**
     * Load the Qwen2 TorchScript model.
     */
    private fun loadModel(context: Context) {
        try {
            Log.d(TAG, "Loading Qwen2 model...")
            val startTime = System.currentTimeMillis()
            
            val modelLoader = ModelLoader(context)
            val modelPath = modelLoader.loadModel("llm_qwen_0.5b_mobile.ptl")
            
            // Load PyTorch Mobile model
            model = LiteModuleLoader.load(modelPath)
            
            val elapsed = System.currentTimeMillis() - startTime
            Log.d(TAG, "Qwen2 model loaded in ${elapsed}ms")
            Log.d(TAG, "Vocabulary size: ${tokenizer.vocabSize()}")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load LLM model", e)
            throw e
        }
    }
    
    /**
     * Predict the next word given a text prompt.
     * 
     * @param prompt Input text (e.g., "Could you please")
     * @return Predicted next word (e.g., " provide")
     */
    suspend fun predictNextWord(prompt: String): String = withContext(Dispatchers.Default) {
        try {
            val startTime = System.currentTimeMillis()
            
            // 1. Tokenize input
            Log.d(TAG, "Tokenizing prompt: '$prompt'")
            val tokenIds = tokenizer.encode(prompt)
            Log.d(TAG, "Token IDs: ${tokenIds.joinToString(", ")}")
            
            if (tokenIds.isEmpty()) {
                Log.w(TAG, "Empty token sequence")
                return@withContext ""
            }
            
            // 2. Run model inference
            Log.d(TAG, "Running LLM inference...")
            val inferenceStart = System.currentTimeMillis()
            val nextTokenId = runInference(tokenIds)
            val inferenceElapsed = System.currentTimeMillis() - inferenceStart
            Log.d(TAG, "Inference time: ${inferenceElapsed}ms")
            
            // 3. Decode next token
            val nextWord = tokenizer.decodeToken(nextTokenId) ?: ""
            Log.d(TAG, "Predicted token ID: $nextTokenId → '$nextWord'")
            
            val totalElapsed = System.currentTimeMillis() - startTime
            Log.d(TAG, "Total prediction time: ${totalElapsed}ms")
            
            nextWord
            
        } catch (e: Exception) {
            Log.e(TAG, "Prediction failed", e)
            throw e
        }
    }
    
    /**
     * Run model inference to get next token logits.
     * 
     * @param tokenIds Input token sequence
     * @return Predicted next token ID
     */
    private fun runInference(tokenIds: IntArray): Int {
        val currentModel = model ?: throw IllegalStateException("Model not loaded")
        
        // Truncate if too long
        val input = if (tokenIds.size > MAX_LENGTH) {
            tokenIds.takeLast(MAX_LENGTH).toIntArray()
        } else {
            tokenIds
        }
        
        // Create input tensor: (1, seq_len) - batch size 1
        val inputShape = longArrayOf(1, input.size.toLong())
        val inputTensor = Tensor.fromBlob(
            input.map { it.toLong() }.toLongArray(),
            inputShape
        )
        
        // Run forward pass
        val output = currentModel.forward(IValue.from(inputTensor)).toTensor()
        
        // Output shape: (1, seq_len, vocab_size)
        val outputData = output.dataAsFloatArray
        
        // Get logits for the last token: [:, -1, :]
        val seqLen = input.size
        val lastTokenLogits = FloatArray(VOCAB_SIZE) { i ->
            // Index: batch=0, position=seqLen-1, vocab=i
            outputData[(seqLen - 1) * VOCAB_SIZE + i]
        }
        
        // Find token with highest logit (greedy decoding)
        var maxIdx = 0
        var maxVal = lastTokenLogits[0]
        for (i in 1 until VOCAB_SIZE) {
            if (lastTokenLogits[i] > maxVal) {
                maxVal = lastTokenLogits[i]
                maxIdx = i
            }
        }
        
        Log.d(TAG, "Max logit: $maxVal at token $maxIdx")
        
        return maxIdx
    }
    
    /**
     * Predict multiple next words (beam search or sampling).
     * For MVP, we just do greedy prediction repeatedly.
     * 
     * @param prompt Input text
     * @param numWords Number of words to predict
     * @return Predicted text continuation
     */
    suspend fun predictNextWords(prompt: String, numWords: Int = 1): String = withContext(Dispatchers.Default) {
        var currentPrompt = prompt
        val predictions = mutableListOf<String>()
        
        repeat(numWords) {
            val nextWord = predictNextWord(currentPrompt)
            if (nextWord.isEmpty()) {
                return@withContext predictions.joinToString("")
            }
            predictions.add(nextWord)
            currentPrompt += nextWord
        }
        
        predictions.joinToString("")
    }
    
    /**
     * Get prediction confidence (softmax probability of top prediction).
     * This helps determine if the suggestion is reliable.
     * 
     * @param prompt Input text
     * @return Confidence score [0.0, 1.0]
     */
    suspend fun getConfidence(prompt: String): Float = withContext(Dispatchers.Default) {
        try {
            val tokenIds = tokenizer.encode(prompt)
            if (tokenIds.isEmpty()) return@withContext 0f
            
            val currentModel = model ?: return@withContext 0f
            
            val input = if (tokenIds.size > MAX_LENGTH) {
                tokenIds.takeLast(MAX_LENGTH).toIntArray()
            } else {
                tokenIds
            }
            
            val inputShape = longArrayOf(1, input.size.toLong())
            val inputTensor = Tensor.fromBlob(
                input.map { it.toLong() }.toLongArray(),
                inputShape
            )
            
            val output = currentModel.forward(IValue.from(inputTensor)).toTensor()
            val outputData = output.dataAsFloatArray
            
            val seqLen = input.size
            val lastTokenLogits = FloatArray(VOCAB_SIZE) { i ->
                outputData[(seqLen - 1) * VOCAB_SIZE + i]
            }
            
            // Apply softmax to get probabilities
            val expLogits = lastTokenLogits.map { exp(it.toDouble()).toFloat() }
            val sumExp = expLogits.sum()
            val probabilities = expLogits.map { it / sumExp }
            
            // Return max probability as confidence
            probabilities.maxOrNull() ?: 0f
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to compute confidence", e)
            0f
        }
    }
    
    /**
     * Clean up resources.
     */
    fun close() {
        Log.d(TAG, "Closing LLM session")
        model?.destroy()
        model = null
    }
    
    private fun exp(x: Double): Double = kotlin.math.exp(x)
}

