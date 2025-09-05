package com.looperpedal.app

import android.Manifest
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Bundle
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.core.content.ContextCompat

class LooperPedalActivity : ComponentActivity() {

    // Unique IDs for all UI elements
    private val viewIds = (1..50).map { View.generateViewId() }.toMutableList()

    companion object {
        init {
            // This loads the shared library "liblooper.so" built from your C++ code
            System.loadLibrary("looper")
        }
    }

    // Declare native JNI functions. The function names must match the Java_..._MethodName
    // format defined in your C++ jni_bridge.cpp file.
    external fun initEngine(sampleRate: Float)
    external fun handleMainButton(trackIndex: Int)
    external fun handleStopButton(trackIndex: Int)
    external fun clearTrack(trackIndex: Int)
    external fun setTrackVolume(trackIndex: Int, volume: Float)
    external fun toggleLiveMonitoring(enabled: Boolean)
    external fun addEffectToTrack(trackIndex: Int, fxType: Int)
    external fun removeEffectFromTrack(trackIndex: Int, fxId: Int)
    external fun toggleEffectBypass(trackIndex: Int, fxId: Int)
    external fun setEffectParameter(trackIndex: Int, fxId: Int, paramIndex: Int, value: Float)
    external fun getUIState(): String
    external fun moveEffectInTrack(trackIndex: Int, fxId: Int, direction: Int)
    external fun toggleBeforeFx(fx_id: Int, active: Boolean)
    external fun setBeforeFxParameter(fx_type: Int, param_index: Int, value: Float)

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            initAudioEngine()
        } else {
            // Handle permission denial
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(createLooperUI())

        requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
    }

    private fun initAudioEngine() {
        val sampleRate = AudioTrack.getNativeOutputSampleRate(AudioManager.STREAM_MUSIC).toFloat()
        initEngine(sampleRate)
    }

    /**
     * Creates the entire looper pedal UI programmatically to mirror the HTML layout.
     */
    private fun createLooperUI(): ViewGroup {
        val dp1 = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 1f, resources.displayMetrics).toInt()
        val dp8 = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 8f, resources.displayMetrics).toInt()
        val dp16 = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 16f, resources.displayMetrics).toInt()
        val dp24 = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 24f, resources.displayMetrics).toInt()
        val dp30 = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 30f, resources.displayMetrics).toInt()
        val sp12 = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, 12f, resources.displayMetrics)

        // Root Layout
        val rootLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setBackgroundColor(0xFF191B22.toInt()) // Using a fixed color for simplicity
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }

        // Header Section
        val headerLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).also { it.setMargins(dp16, dp16, dp16, dp16) }
        }

        val titleTextView = TextView(this).apply {
            text = "Looper Pedal Board"
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 24f)
            setTextColor(0xFFF5F6FA.toInt())
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f)
        }

        val bpmLabel = TextView(this).apply {
            text = "BPM: --"
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 18f)
            setTextColor(0xFFA7FFED.toInt())
            gravity = Gravity.CENTER
            setBackgroundColor(0xFF1C2432.toInt())
            setPadding(dp16, dp8, dp16, dp8)
            background = ContextCompat.getDrawable(context, android.R.drawable.dialog_holo_light_frame)
            background?.setTint(0xFF1C2432.toInt())
        }

        val masterRecBtn = Button(this).apply {
            text = "● Record Mix"
            setBackgroundColor(0xFF6B21A8.toInt())
            setTextColor(0xFFF0ABFC.toInt())
            setPadding(dp16, dp8, dp16, dp8)
            setOnClickListener { showNotImplementedDialog("Master Mix Recording") }
        }

        headerLayout.addView(titleTextView)
        headerLayout.addView(bpmLabel)
        headerLayout.addView(masterRecBtn)

        rootLayout.addView(headerLayout)

        // Before-FX Row
        val beforeFxRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).also { it.setMargins(dp8, dp16, dp8, dp16) }
        }

        val fxButtons = listOf(
            "Delay", "Reverb", "Pitch Shift", "Flanger", "EQ (5 Band)"
        )
        for (i in fxButtons.indices) {
            val fxBtn = Button(this).apply {
                text = fxButtons[i]
                setBackgroundColor(0xFF222B33.toInt())
                setTextColor(0xFF9FFFE6.toInt())
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 16f)
                setPadding(dp16, dp8, dp16, dp8)
                setOnClickListener {
                    toggleBeforeFx(i + 1, !isSelected)
                    isSelected = !isSelected
                    val color = if (isSelected) 0xFF21F7A8.toInt() else 0xFF9FFFE6.toInt()
                    setTextColor(color)
                    showNotImplementedDialog("Before-FX controls")
                }
            }
            beforeFxRow.addView(fxBtn)
        }

        rootLayout.addView(beforeFxRow)

        // Pedal Board
        val pedalBoard = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).also { it.setMargins(0, dp24, 0, dp16) }
        }

        for (i in 1..4) {
            pedalBoard.addView(createTrackView(i))
        }

        rootLayout.addView(pedalBoard)

        // Live Mic Monitor Button
        val liveMicBtn = Button(this).apply {
            text = "Live MIC OFF"
            id = viewIds[0]
            setBackgroundColor(0xFF00FFD0.toInt())
            setTextColor(0xFF23263C.toInt())
            setPadding(dp24, dp16, dp24, dp16)
            setOnClickListener {
                isSelected = !isSelected
                text = if (isSelected) "Live MIC ON 🎤" else "Live MIC OFF"
                val color = if (isSelected) 0xFFFDE047.toInt() else 0xFF23263C.toInt()
                setBackgroundColor(if (isSelected) 0xFFFDE047.toInt() else 0xFF00FFD0.toInt())
                setTextColor(color)
                toggleLiveMonitoring(isSelected)
            }
        }
        rootLayout.addView(liveMicBtn)

        return rootLayout
    }

    /**
     * Creates a single looper track view with all its controls.
     */
    private fun createTrackView(trackIndex: Int): ViewGroup {
        val dp1 = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 1f, resources.displayMetrics).toInt()
        val dp8 = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 8f, resources.displayMetrics).toInt()
        val dp16 = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 16f, resources.displayMetrics).toInt()
        val dp24 = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 24f, resources.displayMetrics).toInt()

        val trackLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setBackgroundColor(0xFF23242D.toInt())
            setPadding(dp24, dp24, dp24, dp24)
            layoutParams = LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f
            ).also { it.setMargins(dp8, dp8, dp8, dp8) }
        }

        val trackLabel = TextView(this).apply {
            text = if (trackIndex == 1) "MASTER" else "TRACK $trackIndex"
            setTextColor(0xFFF5F6FA.toInt())
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 18f)
            gravity = Gravity.CENTER
        }

        // This is a placeholder for the SVG progress ring and icon
        val progressPlaceholder = TextView(this).apply {
            text = "▶" // Placeholder for looper icon
            setTextColor(0xFFFFFFFF.toInt())
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 36f)
            gravity = Gravity.CENTER
            setPadding(0, dp24, 0, dp24)
        }

        val stateDisplay = TextView(this).apply {
            text = "Ready"
            setTextColor(0xFFD0E8F7.toInt())
            setBackgroundColor(0xFF161A2C.toInt())
            setPadding(dp16, dp8, dp16, dp8)
        }

        val volLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp8, dp1, dp8, dp1)
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).also { it.setMargins(0, dp8, 0, dp8) }
        }
        val volLabel = TextView(this).apply {
            text = "Vol:"
            setTextColor(0xFFD0FFE1.toInt())
        }
        val volSlider = SeekBar(this).apply {
            min = 0
            max = 120
            progress = 90
            layoutParams = LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f
            )
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                    setTrackVolume(trackIndex, progress / 100.0f)
                }
                override fun onStartTrackingTouch(seekBar: SeekBar?) {}
                override fun onStopTrackingTouch(seekBar: SeekBar?) {}
            })
        }
        val volValue = TextView(this).apply {
            text = "90%"
            setTextColor(0xFF88FFE6.toInt())
        }
        volLayout.addView(volLabel)
        volLayout.addView(volSlider)
        volLayout.addView(volValue)

        val mainBtn = Button(this).apply {
            text = "Record/Overdub"
            setBackgroundColor(0xFF265CFF.toInt())
            setTextColor(0xFFFFFFFF.toInt())
            setOnClickListener { handleMainButton(trackIndex) }
        }

        val stopClearLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        }
        val stopBtn = Button(this).apply {
            text = "Stop"
            setBackgroundColor(0xFFF43F5E.toInt())
            setTextColor(0xFFFFFFFF.toInt())
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f)
            setOnClickListener { handleStopButton(trackIndex) }
        }
        val clearBtn = Button(this).apply {
            text = "Clear"
            setBackgroundColor(0xFFFB3F5E.toInt())
            setTextColor(0xFFFFFFFF.toInt())
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f)
            setOnClickListener { clearTrack(trackIndex) }
        }
        stopClearLayout.addView(stopBtn)
        stopClearLayout.addView(clearBtn)

        val fxMenuBtn = Button(this).apply {
            text = "🎛 FX Menu"
            setBackgroundColor(0xFFFDE047.toInt())
            setTextColor(0xFF2B2341.toInt())
            setOnClickListener { showNotImplementedDialog("FX Menu for Track $trackIndex") }
        }

        trackLayout.addView(trackLabel)
        trackLayout.addView(progressPlaceholder)
        trackLayout.addView(stateDisplay)
        trackLayout.addView(volLayout)
        trackLayout.addView(mainBtn)
        trackLayout.addView(stopClearLayout)
        trackLayout.addView(fxMenuBtn)

        return trackLayout
    }

    private fun showNotImplementedDialog(featureName: String) {
        AlertDialog.Builder(this)
            .setTitle("$featureName Not Implemented")
            .setMessage("This feature has been added to the UI but its full functionality is not yet implemented in the native code.")
            .setPositiveButton("OK") { dialog, _ -> dialog.dismiss() }
            .show()
    }
}
