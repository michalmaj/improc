// include/improc/exceptions.hpp
#pragma once

#include <exception>
#include <filesystem>
#include <format>
#include <string>
#include <opencv2/core.hpp>

namespace improc {

// ---------------------------------------------------------------------------
// Utility: human-readable cv_type name (mirrors FormatTraits::name)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Base
// ---------------------------------------------------------------------------
class Exception : public std::exception {
public:
    explicit Exception(std::string message) : message_(std::move(message)) {}
    const char* what() const noexcept override { return message_.c_str(); }
protected:
    std::string message_;
};

// ---------------------------------------------------------------------------
// FormatError — Image<Format> type mismatch
// ---------------------------------------------------------------------------
class FormatError : public Exception {
public:
    FormatError(std::string expected_format, std::string actual_format,
                std::string context = "")
        : Exception(std::format("FormatError: expected {}, got{}{}",
                    expected_format, " " + actual_format,
                    context.empty() ? "" : ". Context: " + context))
        , expected_format_(std::move(expected_format))
        , actual_format_(std::move(actual_format))
        , context_(std::move(context)) {}

    const std::string& expected_format() const { return expected_format_; }
    const std::string& actual_format()   const { return actual_format_;   }
    const std::string& context()         const { return context_;         }

private:
    std::string expected_format_;
    std::string actual_format_;
    std::string context_;
};

// ---------------------------------------------------------------------------
// ParameterError — invalid argument / missing required parameter
// ---------------------------------------------------------------------------
class ParameterError : public Exception {
public:
    ParameterError(std::string param_name, std::string constraint,
                   std::string context = "")
        : Exception(std::format("ParameterError: '{}' {}{}",
                    param_name, constraint,
                    context.empty() ? "" : ". Context: " + context))
        , param_name_(std::move(param_name))
        , constraint_(std::move(constraint))
        , context_(std::move(context)) {}

    const std::string& param_name()  const { return param_name_; }
    const std::string& constraint()  const { return constraint_;  }
    const std::string& context()     const { return context_;     }

private:
    std::string param_name_;
    std::string constraint_;
    std::string context_;
};

// ---------------------------------------------------------------------------
// IoError and subclasses — file / camera I/O failures
// ---------------------------------------------------------------------------
class IoError : public Exception {
public:
    explicit IoError(std::string message) : Exception(std::move(message)) {}
};

class FileNotFoundError : public IoError {
public:
    explicit FileNotFoundError(const std::filesystem::path& path)
        : IoError(std::format("FileNotFoundError: '{}' not found or inaccessible",
                  path.string()))
        , path_(path) {}

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

class CameraError : public IoError {
public:
    CameraError(int device_id, std::string reason)
        : IoError(std::format("CameraError: device {} — {}",
                  device_id, reason))
        , device_id_(device_id) {}

    int device_id() const { return device_id_; }

private:
    int device_id_;
};

// ---------------------------------------------------------------------------
// ModelError — model loading or inference failure
// ---------------------------------------------------------------------------
class ModelError : public Exception {
public:
    ModelError(const std::filesystem::path& path, std::string reason)
        : Exception(std::format("ModelError: '{}' — {}",
                    path.string(), reason))
        , model_path_(path)
        , reason_(std::move(reason)) {}

    const std::filesystem::path& model_path() const { return model_path_; }
    const std::string&           reason()      const { return reason_;     }

private:
    std::filesystem::path model_path_;
    std::string           reason_;
};

// ---------------------------------------------------------------------------
// DataError — ImageLoader / Dataset failures
// ---------------------------------------------------------------------------
class DataError : public Exception {
public:
    explicit DataError(std::string message) : Exception(std::move(message)) {}
};

// ---------------------------------------------------------------------------
// AugmentError — augmentation pipeline failures
// ---------------------------------------------------------------------------
class AugmentError : public Exception {
public:
    explicit AugmentError(std::string message) : Exception(std::move(message)) {}
};

} // namespace improc
