package com.pocketwhisper.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.pocketwhisper.app.ml.ModelLoader
import com.pocketwhisper.app.ml.VadDetector
import com.pocketwhisper.app.ml.AsrTranscriber
import com.pocketwhisper.app.ml.SimpleLLM
import com.pocketwhisper.app.audio.AudioCaptureService
import com.pocketwhisper.app.audio.TtsService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.random.Random

class MainActivity : ComponentActivity() {
    
    private val MICROPHONE_PERMISSION_CODE = 100
    private var isProcessingTest = false
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Request mic permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                MICROPHONE_PERMISSION_CODE
            )
        }
        
        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    VadTestScreen()
                }
            }
        }
    }
    
    @Composable
    fun VadTestScreen() {
        var status by remember { mutableStateOf("Ready to test VAD") }
        var isModelLoaded by remember { mutableStateOf(false) }
        var vadProbability by remember { mutableStateOf(0f) }
        var isProcessing by remember { mutableStateOf(false) }
        var isRecording by remember { mutableStateOf(false) }
        var audioStatus by remember { mutableStateOf("") }
        var transcribedText by remember { mutableStateOf("") }
        var nextWordSuggestion by remember { mutableStateOf("") }
        var lastSpokenWord by remember { mutableStateOf("") }
        var asrTranscriber by remember { mutableStateOf<AsrTranscriber?>(null) }
        val llm = remember { SimpleLLM() }
        var ttsService by remember { mutableStateOf<TtsService?>(null) }
        var silenceFrames by remember { mutableStateOf(0) }
        var pauseDuration by remember { mutableStateOf(0f) }
        var lastSuggestionTime by remember { mutableStateOf(0L) }
        
        // Demo phrases to cycle through
        val demoContexts = listOf("could you", "how do", "thank you", "let me", "i think", "please", "would you")
        var demoIndex by remember { mutableStateOf(0) }
        
        val scope = rememberCoroutineScope()
        val modelLoader = remember { ModelLoader(this) }
        var vadDetector by remember { mutableStateOf<VadDetector?>(null) }
        val audioCapture = remember { AudioCaptureService() }
        
        val updateStatus = { newStatus: String ->
            status = newStatus
        }
        
        // ASR test function
        fun testAsrModel() {
            scope.launch {
                try {
                    withContext(Dispatchers.IO) {
                        withContext(Dispatchers.Main) {
                            status = "Initializing ASR (360MB model)..."
                        }
                        
                        val asr = AsrTranscriber(this@MainActivity)
                        
                        withContext(Dispatchers.Main) {
                            status = "Running ASR test..."
                        }
                        
                        val success = asr.testModel()
                        
                        withContext(Dispatchers.Main) {
                            status = if (success) {
                                "✅ ASR model test passed!"
                            } else {
                                "❌ ASR test failed - check logs"
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.e("MainActivity", "ASR test error", e)
                    status = "ASR Error: ${e.message}"
                }
            }
        }
        
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            // Compact Header
            Text(
                text = "Pocket Whisper - Model Test",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold
            )
            
            // Status Card
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                            text = "Model Status:",
                        style = MaterialTheme.typography.labelLarge
                    )
                        Text(
                            text = if (isModelLoaded) "✅ Loaded" else "❌ Not loaded",
                            style = MaterialTheme.typography.bodyMedium,
                            color = if (isModelLoaded) Color(0xFF4CAF50) else Color(0xFFFF5252)
                        )
                    }
                    
                    Spacer(modifier = Modifier.height(8.dp))
                    
                    Text(
                        text = status,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
            
            // Load Model Button
            if (!isModelLoaded) {
            Button(
                onClick = {
                    scope.launch {
                        status = "Loading VAD model..."
                        try {
                                val module = withContext(Dispatchers.IO) {
                                    modelLoader.loadModel("vad_model.pt")
                                }
                                vadDetector = VadDetector(module)
                                isModelLoaded = true
                                status = "✅ VAD model loaded successfully! (vad_model.pt)"
                            } catch (e: Exception) {
                                status = "❌ Error: ${e.message}"
                        }
                    }
                },
                modifier = Modifier.fillMaxWidth()
            ) {
                    Text("Load VAD Model")
                }
            }
            
            // VAD Test Section
            if (isModelLoaded && vadDetector != null) {
            Spacer(modifier = Modifier.height(16.dp))
            
                // Speech Probability Display
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.primaryContainer
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
            Text(
                            text = "Speech Probability",
                style = MaterialTheme.typography.titleMedium
            )
            
                        Spacer(modifier = Modifier.height(16.dp))
                        
                        // Circular indicator
                        Box(
                            modifier = Modifier
                                .size(120.dp)
                                .background(
                                    color = when {
                                        vadProbability > 0.7f -> Color(0xFF4CAF50)
                                        vadProbability > 0.3f -> Color(0xFFFFA726)
                                        else -> Color(0xFFEF5350)
                                    },
                                    shape = CircleShape
                                ),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = "%.1f%%".format(vadProbability * 100),
                                style = MaterialTheme.typography.headlineMedium,
                                color = Color.White,
                                fontWeight = FontWeight.Bold
                            )
                        }
                        
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        Text(
                            text = when {
                                vadProbability > 0.7f -> "🎤 Speech Detected"
                                vadProbability > 0.3f -> "🤔 Uncertain"
                                else -> "🔇 Silence"
                            },
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                }
                
                // Model Test Buttons Row
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    // ASR Test Button
                    Button(
                        onClick = {
                            scope.launch {
                                status = "Testing ASR model..."
                                testAsrModel()
                            }
                        },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.tertiary
                        )
                    ) {
                        Text("Test ASR", fontSize = 14.sp)
                    }
                }
                
                // Live Transcription Display
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.secondaryContainer
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(12.dp)
                    ) {
                        Text(
                            text = "📝 Live Transcription",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = if (transcribedText.isEmpty()) {
                                "Start recording and speak, then pause..."
                            } else {
                                "\"$transcribedText\" ..."
                            },
                            style = MaterialTheme.typography.bodyMedium,
                            modifier = Modifier.fillMaxWidth(),
                            fontWeight = if (transcribedText.isNotEmpty()) FontWeight.Bold else FontWeight.Normal
                        )
                        
                        // Show next word suggestion if available
                        if (nextWordSuggestion.isNotEmpty()) {
                            Spacer(modifier = Modifier.height(8.dp))
                            Card(
                                colors = CardDefaults.cardColors(
                                    containerColor = MaterialTheme.colorScheme.primaryContainer
                                )
                            ) {
                                Column(
                                    modifier = Modifier.padding(8.dp)
                                ) {
                                    Row {
                                        Text(
                                            text = "🔊 Next word: ",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.primary
                                        )
                                        Text(
                                            text = nextWordSuggestion,
                                            style = MaterialTheme.typography.bodyMedium,
                                            fontWeight = FontWeight.Bold,
                                            color = MaterialTheme.colorScheme.primary
                                        )
                                    }
                                    if (lastSpokenWord.isNotEmpty()) {
                                        Text(
                                            text = "Last spoken: '$lastSpokenWord'",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onPrimaryContainer
                                        )
                                    }
                                }
                            }
                        }
                        
                        // Show pause duration
                        if (pauseDuration > 0.1f) {
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = "Pause: ${String.format("%.1f", pauseDuration)}s",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSecondaryContainer
                            )
                        }
                    }
                }
                
                // Real Audio Section
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.tertiaryContainer
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp)
                    ) {
                        Text(
                            text = "🎤 Real Microphone Test",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        Text(
                            text = audioStatus.ifEmpty { "Speak into the microphone to test with real audio" },
                            style = MaterialTheme.typography.bodySmall
                        )
                        
                        Spacer(modifier = Modifier.height(12.dp))
                        
                        Button(
                            onClick = {
                                if (!isRecording) {
                                    // Check microphone permission first
                                    if (ContextCompat.checkSelfPermission(
                                            this@MainActivity,
                                            Manifest.permission.RECORD_AUDIO
                                        ) != PackageManager.PERMISSION_GRANTED
                                    ) {
                                        ActivityCompat.requestPermissions(
                                            this@MainActivity,
                                            arrayOf(Manifest.permission.RECORD_AUDIO),
                                            MICROPHONE_PERMISSION_CODE
                                        )
                                        audioStatus = "⚠️ Please grant microphone permission"
                                        return@Button
                                    }
                                    
                                    // Start recording
                                    scope.launch {
                                        audioStatus = "Starting microphone..."
                                        
                                        // Monitor audio status
                                        launch {
                                            audioCapture.statusFlow.collectLatest { statusMsg ->
                                                audioStatus = statusMsg
                                            }
                                        }
                                        
                                        // Start audio capture
                                        audioCapture.startCapture(scope)
                                        
                                        // Process audio chunks with VAD
                                        launch {
                                            audioCapture.audioFlow.collectLatest { audioChunk ->
                                                try {
                                                    // Calculate audio level for debugging
                                                    val maxLevel = audioChunk.maxOfOrNull { kotlin.math.abs(it) } ?: 0f
                                                    
                                                    // Amplify quiet audio (but not too much to avoid noise)
                                                    val amplifiedChunk = if (maxLevel < 0.1f && maxLevel > 0.001f) {
                                                        val gain = minOf(5f, 0.3f / maxLevel)  // Amplify to ~30% max
                                                        Log.d("MainActivity", "Amplifying audio by ${gain}x (max level: ${(maxLevel * 100).toInt()}%)")
                                                        FloatArray(audioChunk.size) { i -> 
                                                            audioChunk[i] * gain
                                                        }
                                                    } else {
                                                        audioChunk
                                                    }
                                                    
                                                    val probability = vadDetector?.detectSpeech(amplifiedChunk) ?: 0f
                                                    vadProbability = probability
                                                    
                                                    // Update status based on VAD result
                                                    status = when {
                                                        probability > 0.7f -> "🗣️ SPEECH DETECTED (${(probability * 100).toInt()}%)"
                                                        probability > 0.3f -> "🤔 Possible speech (${(probability * 100).toInt()}%)"
                                                        else -> "🔇 Silence (${(probability * 100).toInt()}%) [max: ${(maxLevel * 100).toInt()}%]"
                                                    }
                                                    
                                                    // Speech detected - simulate transcription for demo
                                                    if (probability > 0.5f) {
                                                        silenceFrames = 0  // Reset pause counter
                                                        nextWordSuggestion = ""  // Clear suggestion while speaking
                                                        
                                                        // Use demo context cycling
                                                        transcribedText = demoContexts[demoIndex % demoContexts.size]
                                                        
                                                    } else if (probability < 0.2f) {
                                                        // Silence detected - count pause
                                                        silenceFrames++
                                                        pauseDuration = silenceFrames * 0.032f  // Each frame is 32ms
                                                        
                                                        // Trigger suggestion after 0.5 second pause
                                                        val now = System.currentTimeMillis()
                                                        if (pauseDuration > 0.5f && 
                                                            pauseDuration < 0.7f &&  // Only trigger once per pause
                                                            transcribedText.isNotEmpty() && 
                                                            ttsService?.isInitialized() == true &&
                                                            (now - lastSuggestionTime) > 2000) {  // Wait 2s between suggestions
                                                            
                                                            // Generate and speak suggestion
                                                            val (nextWord, confidence) = llm.predictNextWord(transcribedText)
                                                            
                                                            if (confidence > 0.5f) {
                                                                nextWordSuggestion = "$nextWord (${(confidence * 100).toInt()}%)"
                                                                
                                                                // Speak the suggestion through phone speaker
                                                                Log.d("MainActivity", "🔊 Speaking suggestion: '$nextWord' for context '$transcribedText'")
                                                                ttsService?.speak(nextWord)
                                                                lastSpokenWord = nextWord
                                                                lastSuggestionTime = now
                                                                
                                                                // Move to next demo context
                                                                demoIndex++
                                                            }
                                                        }
                                                    }
                                                } catch (e: Exception) {
                                                    Log.e("MainActivity", "Error processing audio: ${e.message}", e)
                                                }
                                            }
                                        }
                                        
                                        isRecording = true
                                        
                                        // Initialize ASR if not already done
                                        if (asrTranscriber == null) {
                                            scope.launch(Dispatchers.IO) {
                                                audioStatus = "Loading ASR model..."
                                                asrTranscriber = AsrTranscriber(this@MainActivity)
                                                withContext(Dispatchers.Main) {
                                                    audioStatus = "ASR ready - speak now..."
                                                }
                                            }
                                        }
                                        
                                        // Initialize TTS if not already done
                                        if (ttsService == null) {
                                            scope.launch(Dispatchers.IO) {
                                                audioStatus = "Initializing TTS..."
                                                val tts = TtsService(this@MainActivity)
                                                val success = tts.initialize()
                                                withContext(Dispatchers.Main) {
                                                    if (success) {
                                                        ttsService = tts
                                                        audioStatus = "✅ TTS ready - will speak suggestions"
                                                    } else {
                                                        audioStatus = "⚠️ TTS init failed"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    // Stop recording
                                    audioCapture.stopCapture()
                                    isRecording = false
                                    audioStatus = "Microphone stopped"
                                    status = "Recording stopped"
                                    
                                    // Stop TTS
                                    ttsService?.stop()
                                    
                                    // Clear state
                                    transcribedText = ""
                                    nextWordSuggestion = ""
                                    silenceFrames = 0
                                    pauseDuration = 0f
                                    lastSuggestionTime = 0L
                                    demoIndex = 0
                                }
                            },
                            modifier = Modifier.fillMaxWidth(),
                            enabled = vadDetector != null && !isProcessing,
                            colors = if (isRecording) {
                                ButtonDefaults.buttonColors(
                                    containerColor = MaterialTheme.colorScheme.error
                                )
                            } else {
                                ButtonDefaults.buttonColors()
                            }
                        ) {
                            Text(
                                text = if (isRecording) "⏹️ Stop Recording" else "🎤 Start Recording",
                                fontWeight = FontWeight.Medium
                            )
                        }
                    }
                }
            }
            
            Spacer(modifier = Modifier.weight(1f))
            
            // Info Text
            Text(
                text = "Model: Silero VAD v4 (2.2MB)\n" +
                       "Input: 512 samples @ 16kHz (32ms)\n" +
                       "Output: Speech probability [0-1]",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        
        // Cleanup when composable leaves
        DisposableEffect(Unit) {
            onDispose {
                if (isRecording) {
                    audioCapture.stopCapture()
                }
                ttsService?.shutdown()
            }
        }
    }
    
    private suspend fun testWithSimulatedAudio(
        vadDetector: VadDetector,
        simulateSpeech: Boolean,
        updateStatus: (String) -> Unit,
        onResult: (Float) -> Unit
    ) {
        withContext(Dispatchers.Default) {
            try {
                Log.d("MainActivity", "testWithSimulatedAudio: simulateSpeech=$simulateSpeech")
                
                // Create simulated audio chunk (512 samples)
                val audioChunk = FloatArray(512) { i ->
                    if (simulateSpeech) {
                        // Simulate speech-like signal
                        (Random.nextFloat() - 0.5f) * 0.3f + 
                        kotlin.math.sin(i * 0.05f) * 0.2f
                    } else {
                        // Simulate silence/noise
                        (Random.nextFloat() - 0.5f) * 0.01f
                    }
                }
                
                Log.d("MainActivity", "Created audio chunk with ${audioChunk.size} samples")
                Log.d("MainActivity", "Sample values: ${audioChunk.take(5).joinToString()}")
                
                val probability = vadDetector.detectSpeech(audioChunk)
                
                Log.d("MainActivity", "VAD returned probability: $probability")
                
                withContext(Dispatchers.Main) {
                    onResult(probability)
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "Error in testWithSimulatedAudio: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    updateStatus("Error: ${e.message}")
                    onResult(0.5f)
                }
            }
        }
    }
    
    private suspend fun continuousTest(
        vadDetector: VadDetector,
        onResult: (Float) -> Unit
    ) {
        withContext(Dispatchers.Default) {
            var iteration = 0
            while (isProcessingTest) {
                // Alternate between speech and silence simulation
                val simulateSpeech = (iteration / 10) % 2 == 0
                
                val audioChunk = FloatArray(512) { i ->
                    if (simulateSpeech) {
                        (Random.nextFloat() - 0.5f) * 0.3f + 
                        kotlin.math.sin(i * 0.05f + iteration * 0.1f) * 0.2f
                    } else {
                        (Random.nextFloat() - 0.5f) * 0.01f
                    }
                }
                
                val probability = vadDetector.detectSpeech(audioChunk)
                withContext(Dispatchers.Main) {
                    onResult(probability)
                }
                
                delay(100) // Process every 100ms
                iteration++
            }
        }
    }
}