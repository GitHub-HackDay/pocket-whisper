/**
 * JNI bindings for whisper.cpp
 * 
 * Provides Java/Kotlin interface to the native whisper.cpp library.
 */

#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include "whisper.h"

#define TAG "WhisperJNI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

extern "C" {

/**
 * Initialize whisper context from model file.
 * 
 * @param modelPath Path to ggml model file
 * @return Pointer to whisper context (as long) or 0 on failure
 */
JNIEXPORT jlong JNICALL
Java_com_pocketwhisper_app_ml_WhisperCpp_nativeInit(
    JNIEnv* env,
    jobject thiz,
    jstring modelPath
) {
    const char* model_path = env->GetStringUTFChars(modelPath, nullptr);
    
    LOGD("Initializing whisper context from: %s", model_path);
    
    // Initialize whisper context
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;  // CPU for now, can enable GPU later
    
    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path, cparams);
    
    env->ReleaseStringUTFChars(modelPath, model_path);
    
    if (ctx == nullptr) {
        LOGE("Failed to initialize whisper context");
        return 0;
    }
    
    LOGD("✓ Whisper context initialized successfully");
    return reinterpret_cast<jlong>(ctx);
}

/**
 * Transcribe audio data to text.
 * 
 * @param contextPtr Pointer to whisper context
 * @param audioData Float array of audio samples (16kHz mono)
 * @return Transcribed text
 */
JNIEXPORT jstring JNICALL
Java_com_pocketwhisper_app_ml_WhisperCpp_nativeTranscribe(
    JNIEnv* env,
    jobject thiz,
    jlong contextPtr,
    jfloatArray audioData
) {
    auto* ctx = reinterpret_cast<struct whisper_context*>(contextPtr);
    
    if (ctx == nullptr) {
        LOGE("Invalid whisper context");
        return env->NewStringUTF("");
    }
    
    // Get audio samples
    jsize audio_len = env->GetArrayLength(audioData);
    jfloat* audio = env->GetFloatArrayElements(audioData, nullptr);
    
    LOGD("Transcribing %d samples...", audio_len);
    
    // Set default parameters
    struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.language = "en";
    wparams.n_threads = 4;
    wparams.translate = false;
    wparams.print_realtime = false;
    wparams.print_progress = false;
    wparams.print_timestamps = false;
    wparams.print_special = false;
    wparams.no_context = false;
    wparams.single_segment = false;
    
    // Run transcription
    int result = whisper_full(ctx, wparams, audio, audio_len);
    
    env->ReleaseFloatArrayElements(audioData, audio, JNI_ABORT);
    
    if (result != 0) {
        LOGE("whisper_full failed with code: %d", result);
        return env->NewStringUTF("");
    }
    
    // Get transcription result
    const int n_segments = whisper_full_n_segments(ctx);
    std::string transcription;
    
    for (int i = 0; i < n_segments; i++) {
        const char* text = whisper_full_get_segment_text(ctx, i);
        transcription += text;
    }
    
    LOGD("✓ Transcription complete: %d segments", n_segments);
    
    return env->NewStringUTF(transcription.c_str());
}

/**
 * Transcribe with custom parameters.
 */
JNIEXPORT jstring JNICALL
Java_com_pocketwhisper_app_ml_WhisperCpp_nativeTranscribeWithParams(
    JNIEnv* env,
    jobject thiz,
    jlong contextPtr,
    jfloatArray audioData,
    jobject params
) {
    auto* ctx = reinterpret_cast<struct whisper_context*>(contextPtr);
    
    if (ctx == nullptr) {
        LOGE("Invalid whisper context");
        return env->NewStringUTF("");
    }
    
    // Get audio samples
    jsize audio_len = env->GetArrayLength(audioData);
    jfloat* audio = env->GetFloatArrayElements(audioData, nullptr);
    
    // Get WhisperParams class and fields
    jclass paramsClass = env->GetObjectClass(params);
    jfieldID languageField = env->GetFieldID(paramsClass, "language", "Ljava/lang/String;");
    jfieldID translateField = env->GetFieldID(paramsClass, "translate", "Z");
    jfieldID noContextField = env->GetFieldID(paramsClass, "noContext", "Z");
    jfieldID singleSegmentField = env->GetFieldID(paramsClass, "singleSegment", "Z");
    jfieldID beamSizeField = env->GetFieldID(paramsClass, "beamSize", "I");
    
    // Extract parameters
    auto languageStr = (jstring)env->GetObjectField(params, languageField);
    const char* language = env->GetStringUTFChars(languageStr, nullptr);
    bool translate = env->GetBooleanField(params, translateField);
    bool noContext = env->GetBooleanField(params, noContextField);
    bool singleSegment = env->GetBooleanField(params, singleSegmentField);
    int beamSize = env->GetIntField(params, beamSizeField);
    
    // Set whisper parameters
    struct whisper_full_params wparams = whisper_full_default_params(
        beamSize > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY
    );
    wparams.language = language;
    wparams.n_threads = 4;
    wparams.translate = translate;
    wparams.no_context = noContext;
    wparams.single_segment = singleSegment;
    wparams.beam_search.beam_size = beamSize;
    wparams.print_realtime = false;
    wparams.print_progress = false;
    
    LOGD("Transcribing with: lang=%s, translate=%d, beamSize=%d", 
         language, translate, beamSize);
    
    // Run transcription
    int result = whisper_full(ctx, wparams, audio, audio_len);
    
    env->ReleaseStringUTFChars(languageStr, language);
    env->ReleaseFloatArrayElements(audioData, audio, JNI_ABORT);
    
    if (result != 0) {
        LOGE("whisper_full failed with code: %d", result);
        return env->NewStringUTF("");
    }
    
    // Get transcription result
    const int n_segments = whisper_full_n_segments(ctx);
    std::string transcription;
    
    for (int i = 0; i < n_segments; i++) {
        const char* text = whisper_full_get_segment_text(ctx, i);
        transcription += text;
    }
    
    return env->NewStringUTF(transcription.c_str());
}

/**
 * Transcribe with timestamp information.
 */
JNIEXPORT jobject JNICALL
Java_com_pocketwhisper_app_ml_WhisperCpp_nativeTranscribeWithTimestamps(
    JNIEnv* env,
    jobject thiz,
    jlong contextPtr,
    jfloatArray audioData
) {
    auto* ctx = reinterpret_cast<struct whisper_context*>(contextPtr);
    
    if (ctx == nullptr) {
        LOGE("Invalid whisper context");
        return nullptr;
    }
    
    // Get audio samples
    jsize audio_len = env->GetArrayLength(audioData);
    jfloat* audio = env->GetFloatArrayElements(audioData, nullptr);
    
    // Set parameters
    struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.language = "en";
    wparams.n_threads = 4;
    
    // Run transcription
    int result = whisper_full(ctx, wparams, audio, audio_len);
    env->ReleaseFloatArrayElements(audioData, audio, JNI_ABORT);
    
    if (result != 0) {
        LOGE("whisper_full failed");
        return nullptr;
    }
    
    // Create ArrayList for segments
    jclass arrayListClass = env->FindClass("java/util/ArrayList");
    jmethodID arrayListInit = env->GetMethodID(arrayListClass, "<init>", "()V");
    jmethodID arrayListAdd = env->GetMethodID(arrayListClass, "add", "(Ljava/lang/Object;)Z");
    jobject segmentsList = env->NewObject(arrayListClass, arrayListInit);
    
    // Get WhisperSegment class
    jclass segmentClass = env->FindClass("com/pocketwhisper/app/ml/WhisperSegment");
    jmethodID segmentInit = env->GetMethodID(segmentClass, "<init>", 
                                              "(Ljava/lang/String;JJZ)V");
    
    // Extract segments
    const int n_segments = whisper_full_n_segments(ctx);
    
    for (int i = 0; i < n_segments; i++) {
        const char* text = whisper_full_get_segment_text(ctx, i);
        int64_t t0 = whisper_full_get_segment_t0(ctx, i);
        int64_t t1 = whisper_full_get_segment_t1(ctx, i);
        
        // Convert to milliseconds
        jlong start_ms = (t0 * 10);
        jlong end_ms = (t1 * 10);
        
        jstring jtext = env->NewStringUTF(text);
        jobject segment = env->NewObject(segmentClass, segmentInit, 
                                        jtext, start_ms, end_ms, JNI_FALSE);
        
        env->CallBooleanMethod(segmentsList, arrayListAdd, segment);
        env->DeleteLocalRef(segment);
        env->DeleteLocalRef(jtext);
    }
    
    return segmentsList;
}

/**
 * Destroy whisper context and free resources.
 */
JNIEXPORT void JNICALL
Java_com_pocketwhisper_app_ml_WhisperCpp_nativeDestroy(
    JNIEnv* env,
    jobject thiz,
    jlong contextPtr
) {
    auto* ctx = reinterpret_cast<struct whisper_context*>(contextPtr);
    
    if (ctx != nullptr) {
        whisper_free(ctx);
        LOGD("✓ Whisper context destroyed");
    }
}

} // extern "C"

