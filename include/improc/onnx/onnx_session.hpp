// include/improc/onnx/onnx_session.hpp
#pragma once

#include <expected>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include "improc/error.hpp"

namespace improc::onnx {

/**
 * @brief Named tensor carrying a shape and flat float data.
 *
 * Used as the exchange type between `OnnxSession` and its callers.
 * Data is stored flat in row-major (C) order matching ONNX conventions.
 */
struct TensorInfo {
    std::string          name;  ///< Input or output node name as declared in the model.
    std::vector<int64_t> shape; ///< Tensor dimensions, e.g. `{1, 3, 224, 224}`.
    std::vector<float>   data;  ///< Flat float values, row-major order.
};

/**
 * @brief Thin ONNX Runtime session wrapper.
 *
 * Loads a single `.onnx` model file and provides a `run()` method that accepts
 * and returns flat `TensorInfo` values. ORT types are fully hidden behind a
 * pimpl — including the header does not require `onnxruntime_cxx_api.h`.
 *
 * On Apple platforms the CoreML execution provider is automatically registered
 * when available, with transparent CPU fallback for unsupported ops.
 *
 * @code
 * OnnxSession session;
 * if (auto err = session.load("model.onnx"); !err) { ... }
 * auto result = session.run({{session.input_names()[0], {1,3,224,224}, data}});
 * @endcode
 */
class OnnxSession {
public:
    /// @brief Constructs an unloaded session.
    OnnxSession();
    ~OnnxSession();

    OnnxSession(const OnnxSession&)            = delete;
    OnnxSession& operator=(const OnnxSession&) = delete;
    OnnxSession(OnnxSession&&) noexcept;
    OnnxSession& operator=(OnnxSession&&) noexcept;

    /**
     * @brief Loads the ONNX model from disk and initialises the ORT session.
     * @param path Path to a `.onnx` model file.
     * @return `{}` on success; an `improc::Error` if the file is missing,
     *         has the wrong extension, or ORT fails to parse it.
     */
    std::expected<void, improc::Error> load(const std::filesystem::path& path);

    /**
     * @brief Runs inference with the provided input tensors.
     * @param inputs One `TensorInfo` per model input, in declaration order.
     * @return Output tensors in model declaration order, or an `improc::Error`
     *         if the session is not loaded or ORT returns an error.
     */
    std::expected<std::vector<TensorInfo>, improc::Error>
    run(const std::vector<TensorInfo>& inputs) const;

    /**
     * @brief Returns the input node names declared by the loaded model.
     * @return Empty vector if the session has not been loaded yet.
     */
    [[nodiscard]] std::vector<std::string> input_names() const;

    /**
     * @brief Returns the output node names declared by the loaded model.
     * @return Empty vector if the session has not been loaded yet.
     */
    [[nodiscard]] std::vector<std::string> output_names() const;

    /**
     * @brief Returns `true` if a model has been successfully loaded.
     */
    [[nodiscard]] bool is_loaded() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace improc::onnx
