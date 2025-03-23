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
- [CUDA](https://developer.nvidia.com/cuda-zone) (for GPU acceleration)
- [Conan 2.0](https://conan.io/) and [vcpkg](https://github.com/microsoft/vcpkg) for dependency management

## Installation
Follow these steps to build and install the toolkit:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/improc.git
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
