package com.pocketwhisper.app.ml

import android.util.Log
import java.util.LinkedList

/**
 * Manages audio buffering for ASR transcription
 * Accumulates chunks until we have enough for transcription
 */
class AsrBufferManager {
    private val TAG = "AsrBufferManager"
    
    // Buffer to accumulate audio chunks
    private val audioBuffer = LinkedList<FloatArray>()
    
    // Minimum samples for transcription (1 second at 16kHz)
    private val MIN_SAMPLES = 16000
    
    // Maximum buffer size (3 seconds)
    private val MAX_SAMPLES = 16000 * 3
    
    private var totalSamples = 0
    
    /**
     * Add audio chunk to buffer
     * @return Full audio buffer if ready for transcription, null otherwise
     */
    fun addChunk(chunk: FloatArray): FloatArray? {
        audioBuffer.add(chunk)
        totalSamples += chunk.size
        
        // Log buffer status periodically
        if (audioBuffer.size % 10 == 0) {
            Log.d(TAG, "Buffer: ${totalSamples} samples (${totalSamples/16000.0f}s)")
        }
        
        // Check if we have enough audio for transcription
        if (totalSamples >= MIN_SAMPLES) {
            // Combine all chunks
            val fullAudio = FloatArray(totalSamples)
            var offset = 0
            for (chunk in audioBuffer) {
                chunk.copyInto(fullAudio, offset)
                offset += chunk.size
            }
            
            // Keep last 0.5 seconds for continuity
            val keepSamples = 8000 // 0.5 seconds
            if (totalSamples > keepSamples) {
                audioBuffer.clear()
                totalSamples = 0
                
                // Keep some overlap for next transcription
                val overlap = fullAudio.sliceArray(fullAudio.size - keepSamples until fullAudio.size)
                audioBuffer.add(overlap)
                totalSamples = overlap.size
            }
            
            return fullAudio
        }
        
        // Remove old audio if buffer is too large
        while (totalSamples > MAX_SAMPLES) {
            val removed = audioBuffer.removeFirst()
            totalSamples -= removed.size
        }
        
        return null
    }
    
    fun reset() {
        audioBuffer.clear()
        totalSamples = 0
        Log.d(TAG, "Buffer reset")
    }
}
