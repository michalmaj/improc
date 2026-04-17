// include/improc/error.hpp
#pragma once

#include <string>

namespace improc {

// Value type used as the error channel in std::expected<T, improc::Error>.
// Used for environmental / runtime errors where the caller should handle the
// failure gracefully (no try/catch needed).
struct Error {
    enum class Code {
        NoImages,           // ImageLoader found no valid images
        EmptyDataset,       // Dataset directory has no class subdirectories
        DirectoryNotFound,  // Requested directory does not exist
        InvalidModelFile,   // Model path invalid or extension unsupported
        CameraUnavailable,  // Camera device could not be opened
        CameraFrameEmpty,   // Camera is open but returned an empty frame
    };

    Code        code;
    std::string message;

    // Factory helpers — rich, context-aware messages
    static Error no_images(const std::string& dir) {
        return {Code::NoImages,
                "No valid images (.jpg, .jpeg, .png) found in: " + dir};
    }
    static Error empty_dataset(const std::string& dir) {
        return {Code::EmptyDataset,
                "Dataset directory contains no class subdirectories: " + dir};
    }
    static Error directory_not_found(const std::string& path) {
        return {Code::DirectoryNotFound,
                "Directory not found or is not a directory: " + path};
    }
    static Error invalid_model_file(const std::string& path, const std::string& reason) {
        return {Code::InvalidModelFile,
                "Invalid model file '" + path + "': " + reason};
    }
    static Error camera_unavailable(int device_id) {
        return {Code::CameraUnavailable,
                "Camera device " + std::to_string(device_id) + " could not be opened"};
    }
    static Error camera_frame_empty(int device_id) {
        return {Code::CameraFrameEmpty,
                "Camera device " + std::to_string(device_id) + " returned an empty frame"};
    }
};

} // namespace improc
