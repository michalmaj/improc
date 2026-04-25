// include/improc/ml/dnn_forward.hpp
#pragma once

#include <string>
#include <unordered_set>
#include <vector>
#include <opencv2/dnn.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::ml {

using improc::core::Image;
using improc::core::BGR;

/**
 * @brief Minimal OpenCV DNN wrapper that returns the raw flattened output tensor.
 *
 * Use this when the model's output format requires custom parsing that is not
 * covered by `DnnClassifier` or `DnnDetector`. Runs `blobFromImage` → `forward`
 * and flattens all output values into a `std::vector<float>`.
 *
 * Supported model formats: `.onnx`, `.pb`, `.caffemodel`, `.weights`, `.t7`, `.net`.
 *
 * @code
 * std::vector<float> features = DnnForward{"encoder.onnx"}(img);
 * @endcode
 */
struct DnnForward {
    /**
     * @brief Loads the model from disk.
     * @param model_path Path to the model file.
     * @throws improc::ModelError if the file does not exist, has an unsupported
     *         extension, or OpenCV fails to parse it.
     */
    explicit DnnForward(std::string model_path);

    /**
     * @brief Sets the spatial dimensions of the input blob.
     * @param w Width in pixels; must be positive.
     * @param h Height in pixels; must be positive.
     * @return Reference to this op for method chaining.
     * @throws improc::ParameterError if either dimension is not positive.
     */
    DnnForward& input_size(int w, int h) {
        if (w <= 0 || h <= 0) throw ParameterError{"input_size", "dimensions must be positive", "DnnForward"};
        input_w_ = w; input_h_ = h; return *this;
    }

    /**
     * @brief Sets the per-channel mean subtracted from the input blob.
     * @param m Mean values as a `cv::Scalar` (B, G, R).
     * @return Reference to this op for method chaining.
     */
    DnnForward& mean(cv::Scalar m) { mean_ = m; return *this; }

    /**
     * @brief Sets the pixel value scaling factor applied to the input blob.
     * @param s Scale factor; must be positive (default: 1/255).
     * @return Reference to this op for method chaining.
     * @throws improc::ParameterError if @p s <= 0.
     */
    DnnForward& scale(float s) {
        if (s <= 0.0f) throw ParameterError{"scale", "must be positive", "DnnForward"};
        scale_ = s; return *this;
    }

    /**
     * @brief Controls whether R and B channels are swapped when building the blob.
     * @param s `true` to swap (BGR → RGB); default is `true`.
     * @return Reference to this op for method chaining.
     */
    DnnForward& swap_rb(bool s) { swap_rb_ = s; return *this; }

    /**
     * @brief Runs a forward pass and returns the flattened output tensor.
     * @param img Input image in BGR format.
     * @return All output values concatenated into a flat `std::vector<float>`.
     */
    std::vector<float> operator()(const Image<BGR>& img) const;

private:
    mutable cv::dnn::Net  net_;
    int                   input_w_ = 224;
    int                   input_h_ = 224;
    cv::Scalar            mean_    = {0, 0, 0};
    float                 scale_   = 1.0f / 255.0f;
    bool                  swap_rb_ = true;

    /// @brief Returns the set of supported model file extensions.
    static const std::unordered_set<std::string>& valid_extensions();
};

} // namespace improc::ml
