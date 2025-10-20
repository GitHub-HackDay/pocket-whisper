cmake_minimum_required(VERSION 3.18.1)

project("whisper_jni")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to whisper.cpp source
set(WHISPER_CPP_DIR ${CMAKE_SOURCE_DIR}/../../../../../whisper.cpp)

# Include whisper.cpp headers
include_directories(${WHISPER_CPP_DIR})

# Add whisper.cpp source files
set(WHISPER_SOURCES
    ${WHISPER_CPP_DIR}/whisper.cpp
    ${WHISPER_CPP_DIR}/ggml.c
)

# Create native library (whisper JNI wrapper)
add_library(
    whisper
    SHARED
    whisper_jni.cpp
    ${WHISPER_SOURCES}
)

# Find Android log library
find_library(log-lib log)

# Link libraries
target_link_libraries(
    whisper
    ${log-lib}
    android
)

# Compiler flags for optimization
target_compile_options(whisper PRIVATE
    -O3
    -march=armv8.4-a+dotprod+fp16
    -ffast-math
    -funroll-loops
    -pthread
)

# Enable NEON on ARM
if(ANDROID_ABI STREQUAL "arm64-v8a")
    target_compile_definitions(whisper PRIVATE
        GGML_USE_ACCELERATE
    )
endif()

