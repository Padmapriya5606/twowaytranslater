package com.isl.translator

import android.content.res.AssetManager
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Loads sign_model.tflite (converted from src/models/cnn_lstm_sign_model.py
 * via src/tflite_conversion/convert_to_tflite.py) and runs inference on a
 * landmark sequence buffer of shape (SEQ_LEN, LANDMARK_DIM).
 *
 * NOTE: SEQ_LEN and LANDMARK_DIM must exactly match
 * src/training/config.py (SEQ_LEN=90, LANDMARK_DIM=258) and the landmark
 * feature order produced by extract_landmarks.py
 * (left_hand[63] | right_hand[63] | face_subset[69]). The Android-side
 * MediaPipe Holistic extraction (TODO, see android_app/README.md) must
 * produce vectors in that exact same order for the model to work correctly.
 */
class TFLiteSignClassifier(assetManager: AssetManager) {

    companion object {
        private const val MODEL_FILE = "sign_model.tflite"
        private const val LABEL_MAP_FILE = "sign_label_map.json"
        const val SEQ_LEN = 90
        const val LANDMARK_DIM = 258
    }

    private val interpreter: Interpreter
    private val indexToLabel: Map<Int, String>

    init {
        interpreter = Interpreter(loadModelFile(assetManager, MODEL_FILE))
        indexToLabel = loadLabelMap(assetManager, LABEL_MAP_FILE)
    }

    private fun loadModelFile(assetManager: AssetManager, filename: String): MappedByteBuffer {
        val fd = assetManager.openFd(filename)
        val inputStream = fd.createInputStream()
        val channel = inputStream.channel
        return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    private fun loadLabelMap(assetManager: AssetManager, filename: String): Map<Int, String> {
        return try {
            val reader = BufferedReader(InputStreamReader(assetManager.open(filename)))
            val json = JSONObject(reader.readText())
            val map = mutableMapOf<Int, String>()
            json.keys().forEach { className ->
                map[json.getInt(className)] = className
            }
            map
        } catch (e: Exception) {
            emptyMap()
        }
    }

    /**
     * @param landmarkSequence shape (SEQ_LEN, LANDMARK_DIM), same feature
     *   layout as extract_landmarks.py's LandmarkExtractor output. Pad with
     *   zeros if fewer than SEQ_LEN frames were captured.
     * @return predicted gloss label and confidence
     */
    fun predict(landmarkSequence: Array<FloatArray>): Pair<String, Float> {
        require(landmarkSequence.size == SEQ_LEN) { "Expected $SEQ_LEN frames, got ${landmarkSequence.size}" }

        val inputBuffer = ByteBuffer
            .allocateDirect(4 * SEQ_LEN * LANDMARK_DIM)
            .order(ByteOrder.nativeOrder())
        for (frame in landmarkSequence) {
            for (value in frame) {
                inputBuffer.putFloat(value)
            }
        }

        val numClasses = indexToLabel.size.coerceAtLeast(1)
        val output = Array(1) { FloatArray(numClasses) }

        interpreter.run(inputBuffer, output)

        val probs = output[0]
        var bestIdx = 0
        var bestProb = probs.getOrElse(0) { 0f }
        for (i in probs.indices) {
            if (probs[i] > bestProb) {
                bestProb = probs[i]
                bestIdx = i
            }
        }

        val label = indexToLabel[bestIdx] ?: "<UNK>"
        return Pair(label, bestProb)
    }

    fun close() {
        interpreter.close()
    }
}
