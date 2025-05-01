# Namespace Structure in improc++

The **improc++** library is designed in a modular way to facilitate extensibility and maintainability. This document provides an overview of the namespace structure used within the library.

## Overview

The library is organized into the following namespaces:
- `improc` (main namespace)
- `improc::core` (basic image operations)
- `improc::pipeline` (pipeline operations using the pipe operator)
- `improc::threading` (multi-threading support)
- `improc::visualization` (visualization utilities)
- `improc::ml` (machine learning integration)
- `improc::cuda` (CUDA/GPU acceleration)
- `improc::io` (camera input, file I/O, video streaming)
- *(planned)* `improc::math` (generic math helpers and backends)

Below is a detailed description of each namespace.

---

## `improc`

**Purpose:**  
The main namespace for the library that groups all modules and functionalities related to image processing.

---

## `improc::core`

**Purpose:**  
Provides basic image operations.

**Key Elements:**
- **`Image`** – The central class representing an image, typically implemented based on `cv::Mat`.

**Methods:**
- `resize(width, height)` – Resize the image.
- `rotate(angle)` – Rotate the image.
- `crop(x, y, width, height)` – Crop the image.
- `to_grayscale()` – Convert the image to grayscale.
- `apply_filter(filterType)` – Apply a specific filter (e.g., Gaussian, median, bilateral).

---

## `improc::pipeline`

**Purpose:**  
Enables the composition of image processing operations in a pipeline style, similar to `std::ranges`.

**Key Elements:**
- **Functors:**
    - **`Resize`** – Resizes an image.
    - **`Rotate`** – Rotates an image.
    - **`ToGrayscale`** – Converts an image to grayscale.

- **Pipe Operator (`|`):**  
  The library overloads the `|` operator to allow chaining operations in a clear manner.  
  For example:
  ```cpp
  auto processed = image
      | improc::pipeline::Resize(800, 600)
      | improc::pipeline::ToGrayscale();

---

## `improc::threading`

**Purpose:**
Handles asynchronous tasks and multi-threading.

**Key Elements:**
- **`ThreadPool`** – A class for managing and executing asynchronous tasks.

**Methods:**
- **`enqueue`**(task: Function) – Add a task to the queue.
- **`wait()`** – Wait for all tasks to complete.

---

## `improc::visualization`

**Purpose:**
Facilitates the visualization of image processing results.

**Key Elements:**
- **`Plotter`** – A utility class that provides functions to:
    - Plot image histograms.
    - Compare images.
    - Export charts (e.g., to PNG or JPG files).
---

## `improc::ml`

**Purpose:**
Integrates machine learning functionalities for image processing.

**Key Elements:**
- **`FeatureExtractor`** – Provides methods to extract image   features (e.g., HOG, SIFT, ORB).
- **`Classifier`** – Enables training machine learning models (e.g., SVM) and predicting image classes.

---

## `improc::cuda`

**Purpose:**
Provides GPU-accelerated image processing operations using CUDA.

**Key Elements:**
- **`CUDAKernel`** – Implements GPU-based operations, possibly leveraging libraries like thrust or cuBLAS.

---

## `improc::io`

**Purpose:**
Handles input/output operations related to image and video streams.

**Key Elements:**
- CameraCapture – Captures video frames asynchronously from a camera device.
- (planned) VideoReader / ImageLoader – For reading image or video files.

---

## **`(planned) improc::math`**

**Purpose:**
Provides unified math utilities, wrappers, or adapters over different numerical backends (e.g., xtensor, Eigen, Armadillo).

**Use Case Ideas:**
- Matrix conversion utilities
- Abstracted math ops used by `ml` or `cuda` modules