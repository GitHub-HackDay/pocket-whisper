package com.pocketwhisper.app.ml

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * Whisper tokenizer with REAL BPE encoding/decoding.
 * 
 * Uses proper Byte Pair Encoding algorithm for production-ready tokenization.
 * Handles special tokens like <|startoftranscript|>, <|endoftext|>, etc.
 */
class WhisperTokenizer(context: Context) {
    
    private val TAG = "WhisperTokenizer"
    
    // Real BPE tokenizer
    private val bpeTokenizer = BpeTokenizer(context, "asr_tokenizer")
    
    // Special token IDs (Whisper-specific)
    val SOT_TOKEN = 50258 // <|startoftranscript|>
    val EOT_TOKEN = 50257 // <|endoftext|>
    val TRANSCRIBE_TOKEN = 50359 // <|transcribe|>
    val NO_TIMESTAMPS_TOKEN = 50363 // <|notimestamps|>
    val ENGLISH_TOKEN = 50259 // <|en|>
    
    init {
        Log.d(TAG, "WhisperTokenizer initialized with ${bpeTokenizer.vocabSize()} tokens")
    }
    
    /**
     * Decode token IDs to text (with REAL BPE decoding).
     * 
     * @param tokenIds Array of token IDs from the model
     * @param skipSpecialTokens Whether to skip special tokens
     * @return Decoded text string
     */
    fun decode(tokenIds: IntArray, skipSpecialTokens: Boolean = true): String {
        return bpeTokenizer.decode(tokenIds, skipSpecialTokens)
    }
    
    /**
     * Decode a single token ID.
     * 
     * @param tokenId Token ID
     * @return Token string or null if unknown
     */
    fun decodeToken(tokenId: Int): String? {
        return bpeTokenizer.decodeToken(tokenId)
    }
    
    /**
     * Encode text to token IDs (with REAL BPE encoding).
     * 
     * @param text Input text
     * @return Array of token IDs
     */
    fun encode(text: String): IntArray {
        return bpeTokenizer.encode(text)
    }
    
    /**
     * Get the vocabulary size.
     */
    fun vocabSize(): Int = bpeTokenizer.vocabSize()
    
    /**
     * Check if a token ID is a special token.
     */
    fun isSpecialToken(tokenId: Int): Boolean {
        return tokenId in setOf(
            SOT_TOKEN,
            EOT_TOKEN,
            TRANSCRIBE_TOKEN,
            NO_TIMESTAMPS_TOKEN,
            ENGLISH_TOKEN
        ) || tokenId >= 50257 || bpeTokenizer.isSpecialToken(tokenId)
    }
}

