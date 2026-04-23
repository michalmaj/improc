cmake_minimum_required(VERSION 3.30)
project(improc__)

if(APPLE)
    string(REPLACE "--as-needed" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
    string(REPLACE "--no-undefined" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
    string(REPLACE "--as-needed" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
    string(REPLACE "--no-undefined" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

# === improc LIBRARY ===
file(GLOB_RECURSE IMPROC_HEADERS CONFIGURE_DEPENDS include/improc/**/*.hpp)
file(GLOB_RECURSE IMPROC_SOURCES CONFIGURE_DEPENDS src/**/*.cpp)

add_library(improc STATIC ${IMPROC_HEADERS} ${IMPROC_SOURCES})

target_include_directories(improc PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(improc PUBLIC
        opencv::opencv
)

if(APPLE)
    target_link_libraries(improc PUBLIC "-framework Accelerate")
elseif(WIN32 OR UNIX)
    find_package(OpenBLAS CONFIG REQUIRED)
    target_link_libraries(improc PUBLIC OpenBLAS::OpenBLAS)
endif()

enable_testing()

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)

add_executable(improc_tests ${TEST_SOURCES})
target_link_libraries(improc_tests PRIVATE improc GTest::gtest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(improc_tests)


# === DEMO ===

# Camera Capture
add_executable(demo_camera_capture examples/io/demo_camera_capture.cpp)
target_link_libraries(demo_camera_capture PRIVATE improc)

# Image Loader
add_executable(demo_image_loader examples/ml/demo_image_loader.cpp)
target_link_libraries(demo_image_loader PRIVATE improc)

# Dataset loader
add_executable(demo_dataset examples/ml/demo_dataset.cpp)
target_link_libraries(demo_dataset PRIVATE improc)

# OpenCV Haarcascade Loader
add_executable(demo_haar_loader examples/ml/demo_haar_cascade_loader.cpp)
target_link_libraries(demo_haar_loader PRIVATE improc)

# Image Wrapper
add_executable(demo_image_wrapper examples/core/demo_image_wrapper.cpp)
target_link_libraries(demo_image_wrapper PRIVATE improc)

# Pipeline
add_executable(demo_pipeline examples/core/demo_pipeline.cpp)
target_link_libraries(demo_pipeline PRIVATE improc)

# Image Ops
add_executable(demo_image_ops examples/core/demo_image_ops.cpp)
target_link_libraries(demo_image_ops PRIVATE improc)

# Filters, morphology, threshold, padding
add_executable(demo_filters examples/core/demo_filters.cpp)
target_link_libraries(demo_filters PRIVATE improc)

# CLAHE, GammaCorrection, BilateralFilter
add_executable(demo_enhance examples/core/demo_enhance.cpp)
target_link_libraries(demo_enhance PRIVATE improc)

# SobelEdge, CannyEdge
add_executable(demo_edge examples/core/demo_edge.cpp)
target_link_libraries(demo_edge PRIVATE improc)

# FindHomography + WarpPerspective demo
add_executable(demo_homography examples/core/demo_homography.cpp)
target_link_libraries(demo_homography PRIVATE improc)

# M1 ops demos
add_executable(demo_warp_affine examples/core/demo_warp_affine.cpp)
target_link_libraries(demo_warp_affine PRIVATE improc)

add_executable(demo_apply_mask examples/core/demo_apply_mask.cpp)
target_link_libraries(demo_apply_mask PRIVATE improc)

add_executable(demo_unsharp_mask examples/core/demo_unsharp_mask.cpp)
target_link_libraries(demo_unsharp_mask PRIVATE improc)

# Threading demos
add_executable(demo_thread_pool examples/threading/demo_thread_pool.cpp)
target_link_libraries(demo_thread_pool PRIVATE improc)

add_executable(demo_frame_pipeline examples/threading/demo_frame_pipeline.cpp)
target_link_libraries(demo_frame_pipeline PRIVATE improc)

# Visualization demo
add_executable(demo_visualization examples/visualization/demo_visualization.cpp)
target_link_libraries(demo_visualization PRIVATE improc)

# DrawBoundingBoxes demo
add_executable(demo_draw examples/visualization/demo_draw.cpp)
target_link_libraries(demo_draw PRIVATE improc)

# DNN Inference demo
add_executable(demo_dnn_inference examples/ml/demo_dnn_inference.cpp)
target_link_libraries(demo_dnn_inference PRIVATE improc)

# Augmentation demo
add_executable(demo_augmentation examples/ml/demo_augmentation.cpp)
target_link_libraries(demo_augmentation PRIVATE improc)

# Video Writer demo
add_executable(demo_video_writer examples/io/demo_video_writer.cpp)
target_link_libraries(demo_video_writer PRIVATE improc)

# Color adjust demo (Brightness, Contrast, WeightedBlend, AlphaBlend, HSV)
add_executable(demo_color_adjust examples/core/demo_color_adjust.cpp)
target_link_libraries(demo_color_adjust PRIVATE improc)

# Image I/O demo (imread / imwrite)
add_executable(demo_image_io examples/io/demo_image_io.cpp)
target_link_libraries(demo_image_io PRIVATE improc)

# Benchmarks (opt-in: cmake -DIMPROC_BENCHMARKS=ON ...)
if(IMPROC_BENCHMARKS)
    include(FetchContent)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.9.1
        GIT_SHALLOW    TRUE
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)

    file(GLOB_RECURSE BENCH_SOURCES CONFIGURE_DEPENDS benchmarks/*.cpp)
    add_executable(improc_benchmarks ${BENCH_SOURCES})
    target_link_libraries(improc_benchmarks PRIVATE improc benchmark::benchmark_main)
endif()
