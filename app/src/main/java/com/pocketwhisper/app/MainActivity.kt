package com.pocketwhisper.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.pocketwhisper.app.audio.ListenForegroundService
import com.pocketwhisper.app.ui.theme.PocketWhisperTheme
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.values.all { it }
        if (allGranted) {
            // Permissions granted
        } else {
            // Show rationale
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        setContent {
            PocketWhisperTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainScreen(
                        onRequestPermissions = { requestPermissions() },
                        onStartService = { startListeningService() },
                        onStopService = { stopListeningService() },
                        onOpenAccessibilitySettings = { openAccessibilitySettings() }
                    )
                }
            }
        }
    }
    
    private fun requestPermissions() {
        val permissions = arrayOf(
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.POST_NOTIFICATIONS
        )
        requestPermissionLauncher.launch(permissions)
    }
    
    private fun startListeningService() {
        val intent = Intent(this, ListenForegroundService::class.java)
        startForegroundService(intent)
    }
    
    private fun stopListeningService() {
        val intent = Intent(this, ListenForegroundService::class.java)
        stopService(intent)
    }
    
    private fun openAccessibilitySettings() {
        val intent = Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
        startActivity(intent)
    }
}

@Composable
fun MainScreen(
    onRequestPermissions: () -> Unit,
    onStartService: () -> Unit,
    onStopService: () -> Unit,
    onOpenAccessibilitySettings: () -> Unit
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    
    var isListening by remember { mutableStateOf(false) }
    
    // Check permissions
    val hasMicPermission = remember {
        ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "Pocket Whisper",
            style = MaterialTheme.typography.headlineLarge
        )
        
        Spacer(modifier = Modifier.height(48.dp))
        
        // Main toggle card
        Card(
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(
                modifier = Modifier.padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = if (isListening) "Listening ON" else "Listening OFF",
                    style = MaterialTheme.typography.headlineMedium
                )
                
                Spacer(modifier = Modifier.height(16.dp))
                
                Switch(
                    checked = isListening,
                    onCheckedChange = { enabled ->
                        if (enabled && hasMicPermission) {
                            onStartService()
                            isListening = true
                        } else if (enabled) {
                            onRequestPermissions()
                        } else {
                            onStopService()
                            isListening = false
                        }
                    },
                    modifier = Modifier.size(80.dp)
                )
                
                if (!hasMicPermission) {
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = "Microphone permission required",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error
                    )
                }
            }
        }
        
        Spacer(modifier = Modifier.height(32.dp))
        
        // Permissions status card
        Card(
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    "Permissions",
                    style = MaterialTheme.typography.titleMedium
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                PermissionRow(
                    name = "Microphone",
                    granted = hasMicPermission
                )
                
                PermissionRow(
                    name = "Accessibility",
                    granted = false  // TODO: Check accessibility service
                )
                
                Spacer(modifier = Modifier.height(16.dp))
                
                Button(
                    onClick = { onRequestPermissions() },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Request Permissions")
                }
                
                Spacer(modifier = Modifier.height(8.dp))
                
                OutlinedButton(
                    onClick = { onOpenAccessibilitySettings() },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Open Accessibility Settings")
                }
            }
        }
        
        Spacer(modifier = Modifier.weight(1f))
        
        // Info text
        Text(
            text = "All processing happens on-device\nNo internet required",
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.padding(16.dp)
        )
    }
}

@Composable
fun PermissionRow(name: String, granted: Boolean) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(name)
        
        Text(
            text = if (granted) "✓ Granted" else "✗ Not Granted",
            color = if (granted) 
                MaterialTheme.colorScheme.primary 
            else 
                MaterialTheme.colorScheme.error
        )
    }
}

