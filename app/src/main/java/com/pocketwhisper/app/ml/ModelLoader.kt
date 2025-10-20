package com.pocketwhisper.app.ml

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream

/**
 * Utility class for loading ExecuTorch .pte models from assets.
 * 
 * ExecuTorch requires models to be accessible via file path,
 * so we copy them from assets to app cache directory.
 */
class ModelLoader(private val context: Context) {
    
    private val TAG = "ModelLoader"
    
    /**
     * Load a model from assets to cache directory.
     * 
     * @param assetName The name of the asset file (e.g., "vad_silero.pte")
     * @return The absolute file path to the cached model
     * @throws Exception if the asset doesn't exist or copying fails
     */
    fun loadModel(assetName: String): String {
        Log.d(TAG, "Loading model: $assetName")
        
        // Copy from assets to cache (ExecuTorch needs file path)
        val cacheFile = File(context.cacheDir, assetName)
        
        if (!cacheFile.exists()) {
            Log.d(TAG, "Copying $assetName to cache...")
            val startTime = System.currentTimeMillis()
            
            try {
                context.assets.open(assetName).use { input ->
                    FileOutputStream(cacheFile).use { output ->
                        input.copyTo(output)
                    }
                }
                
                val elapsed = System.currentTimeMillis() - startTime
                val sizeMB = cacheFile.length() / (1024.0 * 1024.0)
                Log.d(TAG, "Copied ${sizeMB.format(2)} MB in ${elapsed}ms")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to copy model: $assetName", e)
                throw e
            }
        } else {
            val sizeMB = cacheFile.length() / (1024.0 * 1024.0)
            Log.d(TAG, "Model already cached: ${sizeMB.format(2)} MB")
        }
        
        return cacheFile.absolutePath
    }
    
    /**
     * Load a model only if not already cached.
     * 
     * @param assetName The name of the asset file
     * @return The absolute file path to the cached model, or null if not found
     */
    fun loadModelIfExists(assetName: String): String? {
        return try {
            loadModel(assetName)
        } catch (e: Exception) {
            Log.w(TAG, "Model not found: $assetName")
            null
        }
    }
    
    /**
     * Clear cached models to free up space.
     * Call this when models need to be updated.
     */
    fun clearCache() {
        Log.d(TAG, "Clearing model cache...")
        val pteFiles = context.cacheDir.listFiles { file -> 
            file.extension == "pte" 
        } ?: emptyArray()
        
        var freedBytes = 0L
        pteFiles.forEach { file ->
            freedBytes += file.length()
            file.delete()
            Log.d(TAG, "Deleted: ${file.name}")
        }
        
        val freedMB = freedBytes / (1024.0 * 1024.0)
        Log.d(TAG, "Cleared ${pteFiles.size} files, freed ${freedMB.format(2)} MB")
    }
    
    private fun Double.format(decimals: Int) = "%.${decimals}f".format(this)
}

