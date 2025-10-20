package com.pocketwhisper.app.ml

import android.content.Context
import android.util.Log
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream

class ModelLoader(private val context: Context) {
    
    private val TAG = "ModelLoader"
    
    /**
     * Load a PyTorch Mobile (.pt) model from assets.
     * Returns a Module ready for inference.
     */
    fun loadModel(assetName: String): Module {
        Log.d(TAG, "Loading PyTorch model: $assetName")
        
        // Copy from assets to cache (PyTorch Mobile requires file path)
        val cacheFile = File(context.cacheDir, assetName)
        
        if (!cacheFile.exists()) {
            Log.d(TAG, "Copying $assetName to cache...")
            context.assets.open(assetName).use { input ->
                FileOutputStream(cacheFile).use { output ->
                    input.copyTo(output)
                }
            }
        }
        
        // Load module using PyTorch Mobile Lite
        val module = LiteModuleLoader.load(cacheFile.absolutePath)
        Log.d(TAG, "✅ Model loaded: $assetName (${cacheFile.length() / 1024}KB)")
        
        return module
    }
    
    /**
     * Helper to create a FloatArray tensor for VAD input
     */
    fun createFloatTensor(data: FloatArray): Tensor {
        return Tensor.fromBlob(data, longArrayOf(1, data.size.toLong()))
    }
    
    /**
     * Helper to extract float value from model output
     */
    fun extractFloatResult(output: IValue): Float {
        return output.toTensor().dataAsFloatArray[0]
    }
}