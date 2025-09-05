#include <jni.h>
#include <string>
#include <android/log.h>
#include <aaudio/AAudio.h>
#include "looperengine.cpp" // Includes the complete looper engine implementation

// Define a log tag for logging messages from C++ to logcat
#define LOG_TAG "LooperNative"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

// A global pointer to hold the single instance of our C++ looper engine.
// This ensures that all JNI calls operate on the same engine object.
static LooperEngine* looperEngine = nullptr;
static aaudio_stream* audioStream = nullptr;

/**
 * The AAudio callback function. It's called by AAudio when it needs more audio data.
 * It passes the call directly to our C++ engine's process method.
 */
static aaudio_data_callback_result_t audioCallback(
        AAudioStream *stream,
        void *userData,
        void *audioData,
        int32_t numFrames) {
    if (looperEngine) {
        // The AAudio stream is set up to provide a float array.
        looperEngine->process(nullptr, static_cast<float*>(audioData), numFrames);
    }
    return AAUDIO_CALLBACK_RESULT_CONTINUE;
}

// All JNI-exposed functions are declared within an extern "C" block to prevent name mangling.
extern "C" {

/**
 * JNI function to initialize the C++ audio engine.
 * It creates a new LooperEngine instance and sets up the AAudio stream.
 */
JNIEXPORT void JNICALL
Java_com_looperpedal_app_LooperPedalActivity_initEngine(JNIEnv *env, jobject /* this */, jfloat sample_rate) {
    if (looperEngine == nullptr) {
        looperEngine = new LooperEngine(sample_rate);
    }

    // AAudio stream setup
    AAudioStreamBuilder *builder;
    AAudio_createStreamBuilder(&builder);
    AAudioStreamBuilder_setDirection(builder, AAUDIO_DIRECTION_OUTPUT);
    AAudioStreamBuilder_setPerformanceMode(builder, AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
    AAudioStreamBuilder_setFormat(builder, AAUDIO_FORMAT_PCM_FLOAT);
    AAudioStreamBuilder_setChannelCount(builder, 2);
    AAudioStreamBuilder_setSampleRate(builder, (int32_t)sample_rate);
    AAudioStreamBuilder_setCallback(builder, audioCallback, nullptr);

    AAudio_createStream(builder, &audioStream);
    AAudioStreamBuilder_delete(builder);

    if (audioStream) {
        aaudio_result_t result = AAudioStream_requestStart(audioStream);
        if (result != AAUDIO_OK) {
            LOGD("Failed to start AAudio stream: %s", AAudio_convertResultToText(result));
        } else {
            LOGD("AAudio stream started successfully.");
        }
    } else {
        LOGD("Failed to create AAudio stream.");
    }
}

/**
 * JNI bridge for the main looper pedal button (Record/Overdub).
 */
JNIEXPORT void JNICALL
Java_com_looperpedal_app_LooperPedalActivity_handleMainButton(JNIEnv *env, jobject /* this */, jint track_index) {
    if (looperEngine) {
        looperEngine->handleMainButton(track_index);
        LOGD("handleMainButton called for track %d", track_index);
    }
}

/**
 * JNI bridge for the stop button.
 */
JNIEXPORT void JNICALL
Java_com_looperpedal_app_LooperPedalActivity_handleStopButton(JNIEnv *env, jobject /* this */, jint track_index) {
    if (looperEngine) {
        looperEngine->handleStopButton(track_index);
        LOGD("handleStopButton called for track %d", track_index);
    }
}

/**
 * JNI bridge for the clear button.
 */
JNIEXPORT void JNICALL
Java_com_looperpedal_app_LooperPedalActivity_clearTrack(JNIEnv *env, jobject /* this */, jint track_index) {
    if (looperEngine) {
        looperEngine->clearTrack(track_index);
        LOGD("clearTrack called for track %d", track_index);
    }
}

/**
 * JNI bridge to set the volume of a specific track.
 */
JNIEXPORT void JNICALL
Java_com_looperpedal_app_LooperPedalActivity_setTrackVolume(JNIEnv *env, jobject /* this */, jint track_index, jfloat volume) {
    if (looperEngine) {
        looperEngine->setTrackVolume(track_index, volume);
        LOGD("setTrackVolume called for track %d with volume %f", track_index, volume);
    }
}

/**
 * JNI bridge to toggle live monitoring of the input signal.
 */
JNIEXPORT void JNICALL
Java_com_looperpedal_app_LooperPedalActivity_toggleLiveMonitoring(JNIEnv *env, jobject /* this */, jboolean enabled) {
    if (looperEngine) {
        looperEngine->setLiveMonitoring(enabled);
        LOGD("toggleLiveMonitoring called with enabled = %d", enabled);
    }
}

/**
 * JNI bridge to add an effect to a track.
 */
JNIEXPORT void JNICALL
Java_com_looperpedal_app_LooperPedalActivity_addEffectToTrack(JNIEnv *env, jobject /* this */, jint track_index, jint fx_type) {
    if (looperEngine) {
        looperEngine->addEffectToTrack(track_index, fx_type);
        LOGD("addEffectToTrack called for track %d with type %d", track_index, fx_type);
    }
}

/**
 * JNI bridge to remove an effect from a track.
 */
JNIEXPORT void JNICALL
Java_com_looperpedal_app_LooperPedalActivity_removeEffectFromTrack(JNIEnv *env, jobject /* this */, jint track_index, jint fx_id) {
    if (looperEngine) {
        looperEngine->removeEffectFromTrack(track_index, fx_id);
        LOGD("removeEffectFromTrack called for track %d with fx_id %d", track_index, fx_id);
    }
}

/**
 * JNI bridge to toggle the bypass state of a track's effect.
 */
JNIEXPORT void JNICALL
Java_com_looperpedal_app_LooperPedalActivity_toggleEffectBypass(JNIEnv *env, jobject /* this */, jint track_index, jint fx_id) {
    if (looperEngine) {
        looperEngine->toggleEffectBypass(track_index, fx_id);
        LOGD("toggleEffectBypass called for track %d with fx_id %d", track_index, fx_id);
    }
}

/**
 * JNI bridge to set a parameter on a track's effect.
 */
JNIEXPORT void JNICALL
Java_com_looperpedal_app_LooperPedalActivity_setEffectParameter(JNIEnv *env, jobject /* this */, jint track_index, jint fx_id, jint param_index, jfloat value) {
    if (looperEngine) {
        looperEngine->setEffectParameter(track_index, fx_id, param_index, value);
        LOGD("setEffectParameter called for track %d, fx_id %d, param_index %d, value %f",
             track_index, fx_id, param_index, value);
    }
}

/**
 * JNI bridge to move an effect within a track's chain.
 */
JNIEXPORT void JNICALL
Java_com_looperpedal_app_LooperPedalActivity_moveEffectInTrack(JNIEnv *env, jobject /* this */, jint track_index, jint fx_id, jint direction) {
    if (looperEngine) {
        looperEngine->moveEffectInTrack(track_index, fx_id, direction);
        LOGD("moveEffectInTrack called for track %d, fx_id %d, direction %d",
             track_index, fx_id, direction);
    }
}

/**
 * JNI bridge for the "Before FX" toggle buttons.
 * This is now implemented and sends a command to the engine.
 */
JNIEXPORT void JNICALL
Java_com_looperpedal_app_LooperPedalActivity_toggleBeforeFx(JNIEnv *env, jobject /* this */, jint fx_id, jboolean active) {
    if (looperEngine) {
        looperEngine->toggleBeforeFx(fx_id, active);
        LOGD("toggleBeforeFx called for fx_id %d with active = %d", fx_id, active);
    }
}

/**
 * JNI bridge to set a parameter on a "Before FX" effect.
 * This is now implemented and sends a command to the engine.
 */
JNIEXPORT void JNICALL
Java_com_looperpedal_app_LooperPedalActivity_setBeforeFxParameter(JNIEnv *env, jobject /* this */, jint fx_type, jint param_index, jfloat value) {
    if (looperEngine) {
        looperEngine->setBeforeFxParameter(fx_type, param_index, value);
        LOGD("setBeforeFxParameter called for fx_type %d, param_index %d, value %f",
             fx_type, param_index, value);
    }
}

/**
 * A placeholder JNI function to get the current UI state.
 * For a production app, this would serialize the engine's state (e.g., BPM, loop states)
 * into a JSON string for the UI to display.
 */
JNIEXPORT jstring JNICALL
Java_com_looperpedal_app_LooperPedalActivity_getUIState(JNIEnv *env, jobject /* this */) {
    if (looperEngine) {
        // In a real app, you would create a C++ function to generate the JSON
        // based on the current state of the engine.
        std::string ui_state_json = "{\"status\":\"pending_implementation\",\"bpm\":\"" + std::to_string((int)looperEngine->masterBPM.load()) + "\"}";
        return env->NewStringUTF(ui_state_json.c_str());
    }
    return env->NewStringUTF("{\"error\":\"engine_not_initialized\"}");
}

} // extern "C"
