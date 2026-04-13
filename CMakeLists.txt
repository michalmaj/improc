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
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

# Armadillo behavior (used by mlpack)
add_compile_definitions(ARMA_DONT_USE_WRAPPER)

# === improc LIBRARY ===
file(GLOB_RECURSE IMPROC_HEADERS CONFIGURE_DEPENDS include/improc/**/*.hpp)
file(GLOB_RECURSE IMPROC_SOURCES CONFIGURE_DEPENDS src/**/*.cpp)

add_library(improc STATIC ${IMPROC_HEADERS} ${IMPROC_SOURCES})

target_include_directories(improc PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(improc PUBLIC
        opencv::opencv
        Eigen3::Eigen
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

# Threading demos
add_executable(demo_thread_pool examples/threading/demo_thread_pool.cpp)
target_link_libraries(demo_thread_pool PRIVATE improc)

add_executable(demo_frame_pipeline examples/threading/demo_frame_pipeline.cpp)
target_link_libraries(demo_frame_pipeline PRIVATE improc)

# Visualization demo
add_executable(demo_visualization examples/visualization/demo_visualization.cpp)
target_link_libraries(demo_visualization PRIVATE improc)