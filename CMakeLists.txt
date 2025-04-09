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
find_package(Matplot++ CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(xtensor REQUIRED)
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
        xtensor
        opencv::opencv
        Eigen3::Eigen
        Matplot++::cimg
        Matplot++::matplot
)

if(APPLE)
    target_link_libraries(improc PUBLIC "-framework Accelerate")
elseif(WIN32 OR UNIX)
    find_package(OpenBLAS CONFIG REQUIRED)
    target_link_libraries(improc PUBLIC OpenBLAS::OpenBLAS)
endif()

# === Executable (demo / main) ===
add_executable(improc__ main.cpp)
target_link_libraries(improc__ PRIVATE improc)

enable_testing()

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)

add_executable(improc_tests ${TEST_SOURCES})
target_link_libraries(improc_tests PRIVATE improc GTest::gtest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(improc_tests)
