package com.pocketwhisper.app.ml

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * Byte Pair Encoding (BPE) tokenizer implementation.
 * 
 * This is a REAL BPE implementation that:
 * - Loads vocabulary and merges from tokenizer files
 * - Implements the BPE algorithm for encoding text to tokens
 * - Handles special tokens properly
 * - Supports both Whisper and Qwen2 tokenization
 * 
 * Reference: https://github.com/openai/gpt-2/blob/master/src/encoder.py
 */
class BpeTokenizer(
    context: Context,
    private val tokenizerPath: String  // e.g., "asr_tokenizer" or "llm_tokenizer"
) {
    
    private val TAG = "BpeTokenizer"
    
    // Vocabulary: token string -> token ID
    private val encoder = mutableMapOf<String, Int>()
    
    // Reverse vocabulary: token ID -> token string
    private val decoder = mutableMapOf<Int, String>()
    
    // BPE merges: pairs of tokens that can be merged
    private val bpeMerges = mutableMapOf<Pair<String, String>, Int>()
    
    // Byte encoder/decoder for handling arbitrary bytes
    private val byteEncoder = createByteEncoder()
    private val byteDecoder = byteEncoder.entries.associate { it.value to it.key }
    
    // Cache for BPE results
    private val bpeCache = mutableMapOf<String, String>()
    
    // Special tokens
    var padTokenId: Int? = null
    var bosTokenId: Int? = null
    var eosTokenId: Int? = null
    var unkTokenId: Int? = null
    
    init {
        loadVocabulary(context)
        loadMerges(context)
    }
    
    /**
     * Create byte-to-unicode mapping for GPT-2 style tokenization.
     * Maps bytes to printable unicode characters.
     */
    private fun createByteEncoder(): Map<Int, Char> {
        val bs = mutableListOf<Int>()
        
        // Printable ASCII
        bs.addAll('!'.code..'~'.code)
        bs.addAll('¡'.code..'¬'.code)
        bs.addAll('®'.code..'ÿ'.code)
        
        val cs = bs.toMutableList()
        var n = 0
        
        // Add non-printable bytes
        for (b in 0..255) {
            if (b !in bs) {
                bs.add(b)
                cs.add(256 + n)
                n++
            }
        }
        
        return bs.zip(cs.map { it.toChar() }).toMap()
    }
    
    /**
     * Load vocabulary from tokenizer files.
     */
    private fun loadVocabulary(context: Context) {
        try {
            Log.d(TAG, "Loading vocabulary from $tokenizerPath...")
            
            // Try vocab.json first
            val vocabFile = "$tokenizerPath/vocab.json"
            val vocabJson = try {
                context.assets.open(vocabFile).use { stream ->
                    BufferedReader(InputStreamReader(stream)).use { it.readText() }
                }
            } catch (e: Exception) {
                // Try tokenizer.json as fallback
                Log.d(TAG, "vocab.json not found, trying tokenizer.json...")
                val tokenizerJson = context.assets.open("$tokenizerPath/tokenizer.json").use { stream ->
                    BufferedReader(InputStreamReader(stream)).use { it.readText() }
                }
                JSONObject(tokenizerJson).getJSONObject("model").getJSONObject("vocab").toString()
            }
            
            val vocab = JSONObject(vocabJson)
            val keys = vocab.keys()
            
            while (keys.hasNext()) {
                val token = keys.next()
                val id = vocab.getInt(token)
                encoder[token] = id
                decoder[id] = token
            }
            
            Log.d(TAG, "Loaded ${encoder.size} tokens")
            
            // Load special tokens from config
            loadSpecialTokens(context)
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load vocabulary", e)
            throw e
        }
    }
    
    /**
     * Load special token IDs from tokenizer_config.json.
     */
    private fun loadSpecialTokens(context: Context) {
        try {
            val configJson = context.assets.open("$tokenizerPath/tokenizer_config.json").use { stream ->
                BufferedReader(InputStreamReader(stream)).use { it.readText() }
            }
            
            val config = JSONObject(configJson)
            
            padTokenId = config.optInt("pad_token_id", -1).takeIf { it >= 0 }
            bosTokenId = config.optInt("bos_token_id", -1).takeIf { it >= 0 }
            eosTokenId = config.optInt("eos_token_id", -1).takeIf { it >= 0 }
            unkTokenId = config.optInt("unk_token_id", -1).takeIf { it >= 0 }
            
            Log.d(TAG, "Special tokens - PAD: $padTokenId, BOS: $bosTokenId, EOS: $eosTokenId, UNK: $unkTokenId")
            
        } catch (e: Exception) {
            Log.w(TAG, "Could not load special tokens", e)
        }
    }
    
    /**
     * Load BPE merges from merges.txt or tokenizer.json.
     */
    private fun loadMerges(context: Context) {
        try {
            Log.d(TAG, "Loading BPE merges...")
            
            // Try merges.txt first (GPT-2 style)
            val merges = try {
                context.assets.open("$tokenizerPath/merges.txt").use { stream ->
                    BufferedReader(InputStreamReader(stream)).useLines { lines ->
                        lines.drop(1) // Skip header
                            .filter { it.isNotBlank() }
                            .toList()
                    }
                }
            } catch (e: Exception) {
                // Try tokenizer.json
                Log.d(TAG, "merges.txt not found, trying tokenizer.json...")
                val tokenizerJson = context.assets.open("$tokenizerPath/tokenizer.json").use { stream ->
                    BufferedReader(InputStreamReader(stream)).use { it.readText() }
                }
                val mergesArray = JSONObject(tokenizerJson)
                    .getJSONObject("model")
                    .getJSONArray("merges")
                
                (0 until mergesArray.length()).map { mergesArray.getString(it) }
            }
            
            // Parse merges: "a b" -> priority
            merges.forEachIndexed { priority, merge ->
                val parts = merge.split(" ")
                if (parts.size == 2) {
                    bpeMerges[Pair(parts[0], parts[1])] = priority
                }
            }
            
            Log.d(TAG, "Loaded ${bpeMerges.size} BPE merges")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load BPE merges", e)
            throw e
        }
    }
    
    /**
     * Get pairs of adjacent elements in a word.
     */
    private fun getPairs(word: List<String>): Set<Pair<String, String>> {
        val pairs = mutableSetOf<Pair<String, String>>()
        for (i in 0 until word.size - 1) {
            pairs.add(Pair(word[i], word[i + 1]))
        }
        return pairs
    }
    
    /**
     * Apply BPE algorithm to a word.
     * This is the core BPE merging algorithm.
     */
    private fun bpe(token: String): String {
        // Check cache
        bpeCache[token]?.let { return it }
        
        var word = token.map { it.toString() }.toMutableList()
        var pairs = getPairs(word)
        
        if (pairs.isEmpty()) {
            return token
        }
        
        while (true) {
            // Find the pair with lowest merge priority (merged first)
            val bigram = pairs.minByOrNull { bpeMerges[it] ?: Int.MAX_VALUE } ?: break
            
            if (bigram !in bpeMerges) {
                break
            }
            
            val (first, second) = bigram
            val newWord = mutableListOf<String>()
            var i = 0
            
            while (i < word.size) {
                val j = word.subList(i, word.size).indexOf(first)
                if (j == -1) {
                    newWord.addAll(word.subList(i, word.size))
                    break
                }
                
                newWord.addAll(word.subList(i, i + j))
                i += j
                
                if (word[i] == first && i < word.size - 1 && word[i + 1] == second) {
                    newWord.add(first + second)
                    i += 2
                } else {
                    newWord.add(word[i])
                    i += 1
                }
            }
            
            word = newWord
            if (word.size == 1) {
                break
            }
            
            pairs = getPairs(word)
        }
        
        val result = word.joinToString(" ")
        bpeCache[token] = result
        return result
    }
    
    /**
     * Encode text to token IDs (REAL BPE encoding).
     */
    fun encode(text: String): IntArray {
        if (text.isEmpty()) return intArrayOf()
        
        val tokens = mutableListOf<Int>()
        
        // Split by whitespace and process each word
        val words = text.split(Regex("\\s+")).filter { it.isNotBlank() }
        
        for (word in words) {
            // Convert to bytes and apply byte encoding
            val byteEncoded = word.toByteArray(Charsets.UTF_8)
                .map { byteEncoder[it.toInt() and 0xFF] ?: '?' }
                .joinToString("")
            
            // Apply BPE
            val bpeTokens = bpe(byteEncoded).split(" ")
            
            // Convert BPE tokens to IDs
            for (bpeToken in bpeTokens) {
                val tokenId = encoder[bpeToken] ?: unkTokenId ?: 0
                tokens.add(tokenId)
            }
        }
        
        return tokens.toIntArray()
    }
    
    /**
     * Decode token IDs to text.
     */
    fun decode(tokenIds: IntArray, skipSpecialTokens: Boolean = true): String {
        val tokens = mutableListOf<String>()
        
        for (id in tokenIds) {
            // Skip special tokens if requested
            if (skipSpecialTokens && isSpecialToken(id)) {
                continue
            }
            
            val token = decoder[id]
            if (token != null) {
                tokens.add(token)
            } else {
                Log.w(TAG, "Unknown token ID: $id")
            }
        }
        
        // Join tokens and decode bytes
        val text = tokens.joinToString("")
        
        // Decode byte encoding back to UTF-8
        val bytes = text.map { byteDecoder[it] ?: 0 }.toByteArray()
        
        return try {
            String(bytes, Charsets.UTF_8)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to decode bytes", e)
            text  // Fallback to raw token string
        }
    }
    
    /**
     * Decode a single token ID.
     */
    fun decodeToken(tokenId: Int): String? {
        return decoder[tokenId]
    }
    
    /**
     * Check if a token ID is a special token.
     */
    fun isSpecialToken(tokenId: Int): Boolean {
        return tokenId == padTokenId ||
               tokenId == bosTokenId ||
               tokenId == eosTokenId ||
               tokenId == unkTokenId
    }
    
    /**
     * Get vocabulary size.
     */
    fun vocabSize(): Int = encoder.size
}

