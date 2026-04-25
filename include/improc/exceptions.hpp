// include/improc/exceptions.hpp
#pragma once

#include <exception>
#include <filesystem>
#include <format>
#include <string>
#include <opencv2/core.hpp>

namespace improc {

/**
 * @brief Maps an OpenCV type constant to a human-readable format name.
 *
 * Returns strings such as `"BGR (CV_8UC3)"` or `"Float32 (CV_32FC1)"`.
 * Unknown types return `"unknown (cv_type=N)"`.
 *
 * @param cv_type An OpenCV type constant (e.g. `CV_8UC3`, `CV_32FC1`).
 * @return Human-readable name for the given type constant.
 */
inline std::string cv_type_to_name(int cv_type) {
    switch (cv_type) {
        case CV_8UC1:  return "Gray (CV_8UC1)";
        case CV_8UC3:  return "BGR (CV_8UC3)";
        case CV_8UC4:  return "BGRA (CV_8UC4)";
        case CV_32FC1: return "Float32 (CV_32FC1)";
        case CV_32FC3: return "Float32C3 (CV_32FC3)";
        default:       return std::format("unknown (cv_type={})", cv_type);
    }
}

/**
 * @brief Root exception for all improc library errors.
 *
 * Catch this type to handle any library-thrown exception in a single handler.
 * Prefer catching the more specific subclasses when possible.
 */
class Exception : public std::exception {
public:
    /**
     * @brief Constructs an Exception with the given message.
     * @param message Human-readable description of the error.
     */
    explicit Exception(std::string message) : message_(std::move(message)) {}
    const char* what() const noexcept override { return message_.c_str(); }
protected:
    std::string message_;
};

/**
 * @brief Thrown when an `Image<Format>` is constructed with a mismatched `cv::Mat` type.
 *
 * Carries the expected format name, the actual format name, and an optional
 * context string identifying where the mismatch was detected.
 */
class FormatError : public Exception {
public:
    /**
     * @brief Constructs a FormatError describing a type mismatch.
     * @param expected_format Name of the expected format (e.g. `"BGR (CV_8UC3)"`).
     * @param actual_format   Name of the format that was actually encountered.
     * @param context         Optional location or operation name for diagnostics.
     */
    FormatError(std::string expected_format, std::string actual_format,
                std::string context = "")
        : Exception(std::format("FormatError: expected {}, got{}{}",
                    expected_format, " " + actual_format,
                    context.empty() ? "" : ". Context: " + context))
        , expected_format_(std::move(expected_format))
        , actual_format_(std::move(actual_format))
        , context_(std::move(context)) {}

    /// @brief Returns the name of the format that was expected.
    const std::string& expected_format() const { return expected_format_; }
    /// @brief Returns the name of the format that was actually encountered.
    const std::string& actual_format()   const { return actual_format_;   }
    /// @brief Returns the optional context string, or an empty string if not set.
    const std::string& context()         const { return context_;         }

private:
    std::string expected_format_;
    std::string actual_format_;
    std::string context_;
};

/**
 * @brief Thrown when an operation parameter is missing, zero, negative, or otherwise invalid.
 *
 * Carries the parameter name, a constraint description, and an optional
 * context string for diagnostics.
 */
class ParameterError : public Exception {
public:
    /**
     * @brief Constructs a ParameterError for an invalid or missing parameter.
     * @param param_name Name of the offending parameter (e.g. `"kernel_size"`).
     * @param constraint Human-readable constraint that was violated (e.g. `"must be positive"`).
     * @param context    Optional location or operation name for diagnostics.
     */
    ParameterError(std::string param_name, std::string constraint,
                   std::string context = "")
        : Exception(std::format("ParameterError: '{}' {}{}",
                    param_name, constraint,
                    context.empty() ? "" : ". Context: " + context))
        , param_name_(std::move(param_name))
        , constraint_(std::move(constraint))
        , context_(std::move(context)) {}

    /// @brief Returns the name of the offending parameter.
    const std::string& param_name()  const { return param_name_; }
    /// @brief Returns the constraint description that was violated.
    const std::string& constraint()  const { return constraint_;  }
    /// @brief Returns the optional context string, or an empty string if not set.
    const std::string& context()     const { return context_;     }

private:
    std::string param_name_;
    std::string constraint_;
    std::string context_;
};

/**
 * @brief Base class for I/O-related exceptions (file and camera errors).
 */
class IoError : public Exception {
public:
    /**
     * @brief Constructs an IoError with the given message.
     * @param message Human-readable description of the I/O failure.
     */
    explicit IoError(std::string message) : Exception(std::move(message)) {}
};

/**
 * @brief Thrown when a required file or directory does not exist or is inaccessible.
 */
class FileNotFoundError : public IoError {
public:
    /**
     * @brief Constructs a FileNotFoundError for the given path.
     * @param path The path that could not be found or accessed.
     */
    explicit FileNotFoundError(const std::filesystem::path& path)
        : IoError(std::format("FileNotFoundError: '{}' not found or inaccessible",
                  path.string()))
        , path_(path) {}

    /// @brief Returns the path that could not be found.
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

/**
 * @brief Thrown when a camera device cannot be opened or fails during capture.
 */
class CameraError : public IoError {
public:
    /**
     * @brief Constructs a CameraError for the given device.
     * @param device_id OpenCV camera device index.
     * @param reason    Human-readable description of the failure.
     */
    CameraError(int device_id, std::string reason)
        : IoError(std::format("CameraError: device {} — {}",
                  device_id, reason))
        , device_id_(device_id) {}

    /// @brief Returns the OpenCV device index of the failing camera.
    int device_id() const { return device_id_; }

private:
    int device_id_;
};

/**
 * @brief Thrown when a model file cannot be loaded or is rejected by OpenCV DNN.
 *
 * Carries the path to the model file and a human-readable reason for the failure.
 */
class ModelError : public Exception {
public:
    /**
     * @brief Constructs a ModelError for the given model file.
     * @param path   Path to the model file that could not be loaded.
     * @param reason Human-readable explanation of the failure.
     */
    ModelError(const std::filesystem::path& path, std::string reason)
        : Exception(std::format("ModelError: '{}' — {}",
                    path.string(), reason))
        , model_path_(path)
        , reason_(std::move(reason)) {}

    /// @brief Returns the path to the model file that could not be loaded.
    const std::filesystem::path& model_path() const { return model_path_; }
    /// @brief Returns the human-readable reason for the load failure.
    const std::string&           reason()      const { return reason_;     }

private:
    std::filesystem::path model_path_;
    std::string           reason_;
};

/**
 * @brief Thrown for data-level failures in `ImageLoader` or `Dataset`.
 */
class DataError : public Exception {
public:
    /**
     * @brief Constructs a DataError with the given message.
     * @param message Human-readable description of the data error.
     */
    explicit DataError(std::string message) : Exception(std::move(message)) {}
};

/**
 * @brief Thrown when an augmentation pipeline precondition fails.
 *
 * Example: calling `OneOf` with no augmentations added.
 */
class AugmentError : public Exception {
public:
    /**
     * @brief Constructs an AugmentError with the given message.
     * @param message Human-readable description of the augmentation failure.
     */
    explicit AugmentError(std::string message) : Exception(std::move(message)) {}
};

} // namespace improc
