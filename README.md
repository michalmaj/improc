# improc++: Modern C++ Image Processing Toolkit

## Overview
improc++ is a modern C++ image processing toolkit that provides advanced image manipulation and analysis features. It supports multi-threading, visualization, machine learning integration, and CUDA acceleration for high performance.

## Features
- **Basic Image Operations:** Resize, rotate, crop, grayscale conversion, and filters (Gaussian, median, bilateral)
- **Multi-threading:** Asynchronous processing and task queuing using `std::thread`/`std::async`
- **Visualization:** Compare images before and after processing, generate image histograms, and export charts (using Matplot++)
- **Machine Learning:** Image normalization, feature extraction (HOG, SIFT, ORB), and image classification with ML Pack
- **CUDA Acceleration:** Custom CUDA kernels and libraries (`thrust`, `cuBLAS`) for faster processing on large images

## Requirements
- C++23 or later
- [OpenCV](https://opencv.org/)
- [Matplot++](https://github.com/alandefreitas/matplotplusplus)
- [ML Pack](https://www.mlpack.org/)
- [xtensor](https://xtensor.readthedocs.io/)
- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
- [GoogleTest](https://github.com/google/googletest)
- [CUDA](https://developer.nvidia.com/cuda-zone) (for GPU acceleration)
- [Conan 2.0](https://conan.io/) and [vcpkg](https://github.com/microsoft/vcpkg) for dependency management

## Dependency Management

- Installed via **Conan**:
    - OpenCV
    - xtensor
    - Eigen
    - GoogleTest (GTest)

- Installed via **vcpkg**:
    - ML Pack
    - Matplot++

## Installation
Follow these steps to build and install the toolkit:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/michalmaj/improc.git
   ```
2. **Install dependencies via Conan:**
   ```bash
   conan install .
   ```
3. **Build the project:**
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```
## CMake Configuration
To configure the project with both Conan and vcpkg in CMake-based environments (e.g., CLion), set the following CMake options:
```bash
-DCMAKE_TOOLCHAIN_FILE=<path_to_vcpkg>/scripts/buildsystems/vcpkg.cmake \
-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="conan_provider.cmake" \
-DCONAN_COMMAND=<path_to_conan_executable>
```
Replace <path_to_vcpkg> and <path_to_conan_executable> with the actual paths on your system.

These settings ensure that:
- vcpkg is used for libraries like Matplot++ and ML Pack
- conan_provider.cmake bootstraps Conan for dependencies like OpenCV, Eigen, and xtensor
- Conan is invoked automatically during CMake configuration

### Sanitizer Configuration (optional)
To enable sanitizers for debugging purposes, you can add the following flags to your CMake configuration:

**Thread Sanitizer**
```bash
-DCMAKE_CXX_FLAGS="-fsanitize=thread -g -O1 -fno-omit-frame-pointer" \
-DCMAKE_C_FLAGS="-fsanitize=thread -g -O1 -fno-omit-frame-pointer" \
-DCMAKE_TOOLCHAIN_FILE=<path_to_vcpkg>/scripts/buildsystems/vcpkg.cmake \
-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="conan_provider.cmake" \
-DCONAN_COMMAND=<path_to_conan_executable>
```

**Address + Undefined Behavior Sanitizer**
```bash
-DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -g -O1 -fno-omit-frame-pointer" \
-DCMAKE_C_FLAGS="-fsanitize=address,undefined -g -O1 -fno-omit-frame-pointer" \
-DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_TOOLCHAIN_FILE=<path_to_vcpkg>/scripts/buildsystems/vcpkg.cmake \
-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="conan_provider.cmake" \
-DCONAN_COMMAND=<path_to_conan_executable>
```

💡 Sanitizers are helpful for detecting threading issues, memory leaks, and undefined behavior during development. Use them with `-O1` and `-g` for best results.

## Tested With

- **C++ Compilers:**
   - GCC 14.2
   - Clang 19.1.7

- **Libraries:**
   - OpenCV: 4.8.1
   - MLPack: 4.5.1
   - Matplot++: 1.2.1
   - xtensor: 0.25.0
   - Eigen: 3.4.0
   - GoogleTest: 1.16.0

*Note:* CUDA and MSVC have not been tested yet.
