package com.pocketwhisper.app.ml

import android.util.Log
import kotlin.random.Random

/**
 * Simple next-word predictor for demo
 * Replace with actual LLM model when ready
 */
class SimpleLLM {
    
    private val TAG = "SimpleLLM"
    
    // Common next words based on context
    private val nextWordMap = mapOf(
        "could you" to listOf("please", "help", "tell", "show", "explain"),
        "i think" to listOf("we", "that", "it's", "the", "about"),
        "thank you" to listOf("for", "very", "so", "again", "everyone"),
        "how do" to listOf("you", "I", "we", "they", "people"),
        "what is" to listOf("the", "your", "this", "that", "a"),
        "can you" to listOf("please", "help", "tell", "show", "explain"),
        "i would" to listOf("like", "love", "appreciate", "prefer", "suggest"),
        "we should" to listOf("consider", "discuss", "think", "try", "make"),
        "let me" to listOf("know", "help", "show", "explain", "check"),
        "do you" to listOf("think", "know", "have", "want", "need"),
        "i need" to listOf("to", "help", "your", "some", "more"),
        "please let" to listOf("me", "us", "them", "her", "him"),
        "would you" to listOf("like", "mind", "please", "be", "consider"),
        "that's a" to listOf("good", "great", "nice", "wonderful", "interesting"),
        "this is" to listOf("a", "the", "my", "your", "our")
    )
    
    // Fallback next words when no context match
    private val commonNextWords = listOf(
        "the", "to", "and", "a", "of", "in", "is", "you", "that", "it",
        "for", "on", "with", "as", "was", "at", "be", "this", "have", "from"
    )
    
    /**
     * Predict next word based on context
     * @param context The last few words spoken
     * @return Predicted next word with confidence
     */
    fun predictNextWord(context: String): Pair<String, Float> {
        Log.d(TAG, "Predicting next word for: '$context'")
        
        if (context.isBlank()) {
            return Pair("", 0f)
        }
        
        // Clean and prepare context
        val cleanContext = context.lowercase().trim()
        val words = cleanContext.split(" ").takeLast(10) // Use last 10 words
        
        // Check for pattern matches (last 2 words)
        if (words.size >= 2) {
            val lastTwo = "${words[words.size - 2]} ${words[words.size - 1]}"
            nextWordMap[lastTwo]?.let { suggestions ->
                val nextWord = suggestions.random()
                val confidence = 0.7f + Random.nextFloat() * 0.2f // 70-90% confidence
                Log.d(TAG, "Pattern match: '$lastTwo' -> '$nextWord' (${confidence*100}%)")
                return Pair(nextWord, confidence)
            }
        }
        
        // Check for single word patterns
        if (words.isNotEmpty()) {
            val lastWord = words.last()
            
            // Special cases
            when (lastWord) {
                "hello", "hi", "hey" -> {
                    val suggestions = listOf("there", "everyone", "folks", "team")
                    return Pair(suggestions.random(), 0.8f)
                }
                "please" -> {
                    val suggestions = listOf("help", "let", "tell", "show", "check")
                    return Pair(suggestions.random(), 0.75f)
                }
                "the" -> {
                    val suggestions = listOf("best", "first", "last", "next", "same")
                    return Pair(suggestions.random(), 0.65f)
                }
            }
        }
        
        // Fallback to common words
        val nextWord = commonNextWords.random()
        val confidence = 0.4f + Random.nextFloat() * 0.2f // 40-60% confidence
        Log.d(TAG, "Fallback prediction: '$nextWord' (${confidence*100}%)")
        
        return Pair(nextWord, confidence)
    }
    
    /**
     * Check if we should trigger a suggestion based on pause and context
     */
    fun shouldSuggest(pauseDuration: Float, lastWords: String): Boolean {
        // Simple heuristic: suggest after 0.5s pause and incomplete phrase
        if (pauseDuration < 0.5f) return false
        
        val words = lastWords.lowercase().split(" ")
        if (words.isEmpty()) return false
        
        // Check for incomplete phrases
        val lastWord = words.last()
        val triggerWords = setOf("the", "a", "to", "please", "could", "would", "should", "can", "let")
        
        return lastWord in triggerWords
    }
}
