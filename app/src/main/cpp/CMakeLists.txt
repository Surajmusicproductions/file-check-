cmake_minimum_required(VERSION 3.22.1)

project(looper)

add_library(looper SHARED
        looperengine.cpp
        jni_bridge.cpp
)

find_library(log-lib log)
find_library(aaudio-lib aaudio)

target_link_libraries(looper
        ${log-lib}
        ${aaudio-lib}
)