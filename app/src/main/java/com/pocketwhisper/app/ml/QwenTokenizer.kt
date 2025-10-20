package com.pocketwhisper.app.ml

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * Qwen2 tokenizer with REAL BPE encoding/decoding.
 * 
 * Uses proper Byte Pair Encoding algorithm for production-ready tokenization.
 * Handles special tokens and full vocabulary.
 */
class QwenTokenizer(context: Context) {
    
    private val TAG = "QwenTokenizer"
    
    // Real BPE tokenizer
    private val bpeTokenizer = BpeTokenizer(context, "llm_tokenizer")
    
    // Special tokens (Qwen2-specific) - will be loaded from tokenizer config
    val PAD_TOKEN_ID get() = bpeTokenizer.padTokenId ?: 151643
    val BOS_TOKEN_ID get() = bpeTokenizer.bosTokenId ?: 151643
    val EOS_TOKEN_ID get() = bpeTokenizer.eosTokenId ?: 151643
    
    init {
        Log.d(TAG, "QwenTokenizer initialized with ${bpeTokenizer.vocabSize()} tokens")
        Log.d(TAG, "Special tokens - PAD: $PAD_TOKEN_ID, BOS: $BOS_TOKEN_ID, EOS: $EOS_TOKEN_ID")
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
     * Decode token IDs to text (with REAL BPE decoding).
     * 
     * @param tokenIds Array of token IDs
     * @param skipSpecialTokens Whether to skip special tokens
     * @return Decoded text
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
     * Get vocabulary size.
     */
    fun vocabSize(): Int = bpeTokenizer.vocabSize()
}

