package com.pocketwhisper.app.ml

import android.util.Log

/**
 * Simple CTC decoder for Wav2Vec2 model
 * Converts logits to text using greedy decoding
 */
class Wav2Vec2Decoder {
    
    private val TAG = "Wav2Vec2Decoder"
    
    // Wav2Vec2 vocabulary (simplified - first 32 chars)
    // Full vocab is 32 chars: <pad> <s> </s> <unk> | E T A O N I H S R D L U M W C F G Y P B V K ' X J Q Z
    private val vocab = listOf(
        "<pad>", "<s>", "</s>", "<unk>", "|",  // Special tokens
        "E", "T", "A", "O", "N", "I", "H", "S", "R", "D",
        "L", "U", "M", "W", "C", "F", "G", "Y", "P", "B",
        "V", "K", "'", "X", "J", "Q", "Z"
    )
    
    /**
     * Decode logits to text using greedy CTC decoding
     * @param logits FloatArray of shape [time_steps * vocab_size]
     * @param timeSteps Number of time steps
     * @param vocabSize Vocabulary size (should be 32 for Wav2Vec2)
     * @return Decoded text
     */
    fun decode(logits: FloatArray, timeSteps: Int, vocabSize: Int): String {
        if (vocabSize != vocab.size) {
            Log.w(TAG, "Vocab size mismatch: expected ${vocab.size}, got $vocabSize")
        }
        
        val tokens = mutableListOf<Int>()
        var prevToken = -1
        
        // Greedy decoding: pick highest probability token at each timestep
        for (t in 0 until timeSteps) {
            val offset = t * vocabSize
            var maxIdx = 0
            var maxVal = logits[offset]
            
            // Find argmax
            for (v in 1 until vocabSize) {
                if (logits[offset + v] > maxVal) {
                    maxVal = logits[offset + v]
                    maxIdx = v
                }
            }
            
            // CTC collapse: skip repeated tokens and blanks
            if (maxIdx != 0 && maxIdx != prevToken) {  // 0 is <pad>/blank
                tokens.add(maxIdx)
            }
            prevToken = maxIdx
        }
        
        // Convert tokens to text
        val words = mutableListOf<String>()
        val currentWord = StringBuilder()
        
        for (token in tokens) {
            if (token >= vocab.size) continue
            
            val char = vocab[token]
            when (char) {
                "|" -> {
                    // Word boundary
                    if (currentWord.isNotEmpty()) {
                        words.add(currentWord.toString())
                        currentWord.clear()
                    }
                }
                "<pad>", "<s>", "</s>", "<unk>" -> {
                    // Skip special tokens
                }
                else -> {
                    currentWord.append(char)
                }
            }
        }
        
        // Add last word
        if (currentWord.isNotEmpty()) {
            words.add(currentWord.toString())
        }
        
        val result = words.joinToString(" ").lowercase()
        Log.d(TAG, "Decoded: '$result' from ${tokens.size} tokens")
        
        return result
    }
    
    /**
     * Quick decode for demo - assumes model outputs are in correct format
     */
    fun quickDecode(logits: FloatArray): String {
        // Auto-detect dimensions
        // Common shapes: [1, time_steps, vocab_size] or [time_steps, vocab_size]
        val totalSize = logits.size
        
        // Assume vocab size of 32 (standard for Wav2Vec2)
        val vocabSize = 32
        val timeSteps = totalSize / vocabSize
        
        if (timeSteps < 1) {
            Log.w(TAG, "Invalid logits size: $totalSize")
            return ""
        }
        
        return decode(logits, timeSteps, vocabSize)
    }
}

