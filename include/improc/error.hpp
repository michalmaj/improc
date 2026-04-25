// include/improc/error.hpp
#pragma once

#include <string>

namespace improc {

/**
 * @brief Value type for the error channel in `std::expected<T, improc::Error>`.
 *
 * Used for environmental and runtime errors where the caller is expected to
 * handle failures gracefully without try/catch. Carries a machine-readable
 * @ref Code and a human-readable @ref message string. Build instances via
 * the named static factory helpers.
 */
struct Error {
    /**
     * @brief Identifies the category of a runtime error.
     */
    enum class Code {
        NoImages,           ///< ImageLoader found no valid images in the directory.
        EmptyDataset,       ///< Dataset directory has no class subdirectories.
        DirectoryNotFound,  ///< Requested path does not exist or is not a directory.
        InvalidModelFile,   ///< Model file path is invalid or has an unsupported extension.
        CameraUnavailable,  ///< Camera device could not be opened.
        CameraFrameEmpty,   ///< Camera is open but returned an empty frame.
        InsufficientPoints, ///< `find_homography`: fewer than 4 or mismatched point pairs.
        HomographyFailed,   ///< `find_homography`: RANSAC returned an invalid homography.
        ImageReadFailed,    ///< `cv::imread` returned an empty mat.
        ImageWriteFailed,      ///< `cv::imwrite` returned false.
        OnnxModelLoadFailed,   ///< ONNX Runtime failed to parse or load the model file.
        OnnxInferenceFailed,   ///< ONNX Runtime session run returned an error.
        OnnxSessionNotLoaded,  ///< `OnnxSession::run()` called before `load()`.
    };

    Code        code;    ///< Machine-readable error category.
    std::string message; ///< Human-readable description of the error.

    /**
     * @brief Returns an error for a directory that contains no loadable images.
     * @param dir Path to the directory that was scanned.
     */
    static Error no_images(const std::string& dir) {
        return {Code::NoImages,
                "No valid images (.jpg, .jpeg, .png) found in: " + dir};
    }

    /**
     * @brief Returns an error for a dataset root with no class subdirectories.
     * @param dir Path to the dataset root directory.
     */
    static Error empty_dataset(const std::string& dir) {
        return {Code::EmptyDataset,
                "Dataset directory contains no class subdirectories: " + dir};
    }

    /**
     * @brief Returns an error for a path that does not exist or is not a directory.
     * @param path The path that was checked.
     */
    static Error directory_not_found(const std::string& path) {
        return {Code::DirectoryNotFound,
                "Directory not found or is not a directory: " + path};
    }

    /**
     * @brief Returns an error for an invalid or unrecognised model file.
     * @param path   Path to the model file.
     * @param reason Human-readable explanation (e.g. `"file not found"`, `"unsupported extension"`).
     */
    static Error invalid_model_file(const std::string& path, const std::string& reason) {
        return {Code::InvalidModelFile,
                "Invalid model file '" + path + "': " + reason};
    }

    /**
     * @brief Returns an error when a camera device cannot be opened.
     * @param device_id OpenCV camera device index.
     */
    static Error camera_unavailable(int device_id) {
        return {Code::CameraUnavailable,
                "Camera device " + std::to_string(device_id) + " could not be opened"};
    }

    /**
     * @brief Returns an error when an open camera returns an empty frame.
     * @param device_id OpenCV camera device index.
     */
    static Error camera_frame_empty(int device_id) {
        return {Code::CameraFrameEmpty,
                "Camera device " + std::to_string(device_id) + " returned an empty frame"};
    }

    /**
     * @brief Returns an error when fewer than 4 corresponding point pairs are provided.
     * @param got Number of point pairs that were actually supplied.
     */
    static Error insufficient_points(std::size_t got) {
        return {Code::InsufficientPoints,
                "find_homography requires equal-length vectors of at least 4 point pairs, got " +
                std::to_string(got)};
    }

    /**
     * @brief Returns an error when RANSAC fails to produce a valid homography.
     */
    static Error homography_failed() {
        return {Code::HomographyFailed,
                "find_homography: RANSAC failed to find a valid homography"};
    }

    /**
     * @brief Returns an error when `cv::imread` returns an empty mat.
     * @param path Path to the image file that failed to load.
     */
    static Error image_read_failed(const std::string& path) {
        return {Code::ImageReadFailed, "Failed to read image: " + path};
    }

    /**
     * @brief Returns an error when `cv::imwrite` fails to save the image.
     * @param path Destination path that was passed to `cv::imwrite`.
     */
    static Error image_write_failed(const std::string& path) {
        return {Code::ImageWriteFailed, "Failed to write image: " + path};
    }

    /**
     * @brief Returns an error when ONNX Runtime fails to load the model.
     * @param path   Path to the `.onnx` file.
     * @param reason Human-readable reason string from the ORT exception.
     */
    static Error onnx_model_load_failed(const std::string& path, const std::string& reason) {
        return {Code::OnnxModelLoadFailed,
                "Failed to load ONNX model '" + path + "': " + reason};
    }

    /**
     * @brief Returns an error when an ONNX Runtime session run fails.
     * @param reason Human-readable reason string from the ORT exception.
     */
    static Error onnx_inference_failed(const std::string& reason) {
        return {Code::OnnxInferenceFailed, "ONNX inference failed: " + reason};
    }

    /**
     * @brief Returns an error when `OnnxSession::run()` is called before `load()`.
     */
    static Error onnx_session_not_loaded() {
        return {Code::OnnxSessionNotLoaded,
                "OnnxSession::run() called before load() — call load() first"};
    }
};

} // namespace improc
