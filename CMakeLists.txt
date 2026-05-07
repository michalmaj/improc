cmake_minimum_required(VERSION 3.30)
project(improc__ VERSION 0.1.0)

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

# Compatibility shim: system OpenCV uses OpenCV_LIBS, Conan uses opencv::opencv
if(NOT TARGET opencv::opencv)
    add_library(opencv::opencv INTERFACE IMPORTED GLOBAL)
    target_link_libraries(opencv::opencv INTERFACE ${OpenCV_LIBS})
    target_include_directories(opencv::opencv INTERFACE ${OpenCV_INCLUDE_DIRS})
endif()

# === ONNX Runtime (optional, CPU + CoreML on Apple) ===
option(IMPROC_WITH_ONNX "Build with ONNX Runtime inference support" ON)

if(IMPROC_WITH_ONNX)
    set(ORT_VERSION "1.20.1")

    if(APPLE)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
            set(_ort_archive "onnxruntime-osx-arm64-${ORT_VERSION}.tgz")
        else()
            set(_ort_archive "onnxruntime-osx-x86_64-${ORT_VERSION}.tgz")
        endif()
    elseif(UNIX)
        set(_ort_archive "onnxruntime-linux-x64-${ORT_VERSION}.tgz")
    else()
        message(FATAL_ERROR "IMPROC_WITH_ONNX: unsupported platform")
    endif()

    include(FetchContent)
    FetchContent_Declare(onnxruntime
        URL      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${_ort_archive}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(onnxruntime)

    # The archive extracts with a named top-level directory inside SOURCE_DIR
    file(GLOB _ort_candidates LIST_DIRECTORIES true "${onnxruntime_SOURCE_DIR}/onnxruntime-*")
    if(_ort_candidates)
        list(GET _ort_candidates 0 ORT_ROOT)
    else()
        set(ORT_ROOT "${onnxruntime_SOURCE_DIR}")
    endif()

    find_library(ORT_LIBRARY NAMES onnxruntime
        PATHS "${ORT_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

    add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION             "${ORT_LIBRARY}"
        IMPORTED_NO_SONAME            TRUE
        INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include"
    )

    message(STATUS "ONNX Runtime ${ORT_VERSION}: ${ORT_LIBRARY}")
endif()

# === improc LIBRARY ===
file(GLOB_RECURSE IMPROC_HEADERS CONFIGURE_DEPENDS include/improc/**/*.hpp)
file(GLOB_RECURSE IMPROC_SOURCES CONFIGURE_DEPENDS src/**/*.cpp)

add_library(improc STATIC ${IMPROC_HEADERS} ${IMPROC_SOURCES})

target_include_directories(improc PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(improc PUBLIC
        opencv::opencv
        $<$<BOOL:${IMPROC_WITH_ONNX}>:onnxruntime::onnxruntime>
)
target_compile_definitions(improc PUBLIC
        $<$<BOOL:${IMPROC_WITH_ONNX}>:IMPROC_WITH_ONNX>
)

enable_testing()

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)

add_executable(improc_tests ${TEST_SOURCES})
target_link_libraries(improc_tests PRIVATE improc GTest::gtest GTest::gtest_main)

if(IMPROC_WITH_ONNX)
    list(APPEND CMAKE_BUILD_RPATH "${ORT_ROOT}/lib")
    set_property(TARGET improc_tests APPEND PROPERTY BUILD_RPATH "${ORT_ROOT}/lib")
endif()

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

# CenterCrop + LetterBox demo
add_executable(demo_geometric_preprocessing examples/core/demo_geometric_preprocessing.cpp)
target_link_libraries(demo_geometric_preprocessing PRIVATE improc)

# AdaptiveThreshold demo
add_executable(demo_adaptive_threshold examples/core/demo_adaptive_threshold.cpp)
target_link_libraries(demo_adaptive_threshold PRIVATE improc)

# Invert + InRange demo
add_executable(demo_invert_in_range examples/core/demo_invert_in_range.cpp)
target_link_libraries(demo_invert_in_range PRIVATE improc)

# MorphOpen + MorphClose demo
add_executable(demo_morph_open_close examples/core/demo_morph_open_close.cpp)
target_link_libraries(demo_morph_open_close PRIVATE improc)

# HistogramEqualization + NLMeansDenoising demo
add_executable(demo_hist_eq_nlmeans examples/core/demo_hist_eq_nlmeans.cpp)
target_link_libraries(demo_hist_eq_nlmeans PRIVATE improc)

# MorphGradient + TopHat + BlackHat demo
add_executable(demo_morph_extras examples/core/demo_morph_extras.cpp)
target_link_libraries(demo_morph_extras PRIVATE improc)

# LaplacianEdge + HarrisCorner demo
add_executable(demo_harris_laplacian examples/core/demo_harris_laplacian.cpp)
target_link_libraries(demo_harris_laplacian PRIVATE improc)

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

# Video Reader demo
add_executable(demo_video_reader examples/io/demo_video_reader.cpp)
target_link_libraries(demo_video_reader PRIVATE improc)

# Montage demo
add_executable(demo_montage examples/visualization/demo_montage.cpp)
target_link_libraries(demo_montage PRIVATE improc)

# Pyramid + Drawing demo
add_executable(demo_pyramid_drawing examples/core/demo_pyramid_drawing.cpp)
target_link_libraries(demo_pyramid_drawing PRIVATE improc)

# Contours demo
add_executable(demo_contours examples/core/demo_contours.cpp)
target_link_libraries(demo_contours PRIVATE improc)

# Connected Components + Distance Transform demo
add_executable(demo_connected_components examples/core/demo_connected_components.cpp)
target_link_libraries(demo_connected_components PRIVATE improc)

# Feature Detection demo
add_executable(demo_feature_detection examples/core/demo_feature_detection.cpp)
target_link_libraries(demo_feature_detection PRIVATE improc)

# Descriptor Computation demo
add_executable(demo_describe examples/core/demo_describe.cpp)
target_link_libraries(demo_describe PRIVATE improc)

# Matching demo
add_executable(demo_matching examples/core/demo_matching.cpp)
target_link_libraries(demo_matching PRIVATE improc)

# DrawKeypoints / DrawMatches demo
add_executable(demo_draw_matches examples/core/demo_draw_matches.cpp)
target_link_libraries(demo_draw_matches PRIVATE improc)

# Views demos (M1: single-image lazy pipeline, M2: collection pipeline, M3: external sources, M4: batch/enumerate/zip)
add_executable(demo_views_m1 examples/views/demo_views_m1.cpp)
target_link_libraries(demo_views_m1 PRIVATE improc)

add_executable(demo_views_m2 examples/views/demo_views_m2.cpp)
target_link_libraries(demo_views_m2 PRIVATE improc)

add_executable(demo_views_m3 examples/views/demo_views_m3.cpp)
target_link_libraries(demo_views_m3 PRIVATE improc)

add_executable(demo_views_m4 examples/views/demo_views_m4.cpp)
target_link_libraries(demo_views_m4 PRIVATE improc)

# ONNX inference demo
if(IMPROC_WITH_ONNX)
    add_executable(demo_onnx_inference examples/ml/demo_onnx_inference.cpp)
    target_link_libraries(demo_onnx_inference PRIVATE improc)
    set_property(TARGET demo_onnx_inference APPEND PROPERTY BUILD_RPATH "${ORT_ROOT}/lib")
endif()

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
