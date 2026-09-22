package com.isl.translator

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.Locale

/**
 * Skeleton activity. Two modes:
 *   - Sign -> Speech: CameraX preview feeds frames to a MediaPipe Holistic
 *     landmark extractor (TODO, see android_app/README.md), buffered
 *     landmark sequences go to [TFLiteSignClassifier], the predicted gloss
 *     is spoken with [TextToSpeech].
 *   - Speech -> Sign: Android's SpeechRecognizer captures the hearing
 *     person's speech, maps it to gloss tokens, and drives the avatar
 *     renderer (TODO — port the keyframe format from
 *     src/avatar/avatar_renderer.py to a Canvas/SurfaceView).
 *
 * This file wires the pieces together and requests permissions; the two
 * TODOs above are the remaining implementation work called out in
 * android_app/README.md.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var statusText: TextView
    private lateinit var tts: TextToSpeech
    private var signClassifier: TFLiteSignClassifier? = null

    private val requestPermissionLauncher =
        registerForActivityResult(
            androidx.activity.result.contract.ActivityResultContracts.RequestMultiplePermissions()
        ) { grants ->
            val allGranted = grants.values.all { it }
            statusText.text = if (allGranted) {
                "Permissions granted. Ready."
            } else {
                "Camera + microphone permissions are required for this app to work."
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.statusText)
        val signToSpeechBtn = findViewById<Button>(R.id.signToSpeechButton)
        val speechToSignBtn = findViewById<Button>(R.id.speechToSignButton)

        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts.language = Locale.US
            }
        }

        try {
            signClassifier = TFLiteSignClassifier(assets)
            statusText.text = "Sign model loaded. Grant permissions to begin."
        } catch (e: Exception) {
            statusText.text = "sign_model.tflite not found in assets/ yet — " +
                "train it and run tflite_conversion, then copy it in. See android_app/README.md."
        }

        signToSpeechBtn.setOnClickListener {
            // TODO: start CameraX preview -> MediaPipe Holistic -> landmark buffer
            // -> signClassifier.predict(sequence) -> tts.speak(result)
            statusText.text = "Sign -> Speech mode selected (camera pipeline: TODO, see README)."
        }

        speechToSignBtn.setOnClickListener {
            // TODO: start SpeechRecognizer -> sentence_to_gloss mapping ->
            // AvatarView.play(glossSequence)
            statusText.text = "Speech -> Sign mode selected (STT + avatar pipeline: TODO, see README)."
        }

        requestPermissionLauncher.launch(
            arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
        )
    }

    override fun onDestroy() {
        tts.shutdown()
        signClassifier?.close()
        super.onDestroy()
    }
}
