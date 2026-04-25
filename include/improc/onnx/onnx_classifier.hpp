// include/improc/onnx/onnx_classifier.hpp
#pragma once

#include <expected>
#include <string>
#include <vector>
#include "improc/core/image.hpp"
#include "improc/ml/result_types.hpp"
#include "improc/error.hpp"
#include "improc/exceptions.hpp"
#include "improc/onnx/onnx_session.hpp"

namespace improc::onnx {

using improc::core::Image;
using improc::core::BGR;
using improc::ml::ClassResult;

/**
 * @brief ONNX Runtime-backed image classifier.
 *
 * Wraps `OnnxSession` with a full image-to-`ClassResult` pipeline:
 * resize → float conversion → mean subtraction → optional channel swap →
 * HWC→CHW → inference → top-k selection.
 *
 * Expects the model's first input to be a float32 tensor of shape
 * `[1, C, H, W]` and the first output to be a 1-D score vector `[1, N]`.
 * Fluent setters mirror the `DnnClassifier` API for consistency.
 *
 * @code
 * auto cls = OnnxClassifier{"mobilenet.onnx"}.input_size(224, 224).top_k(3);
 * auto results = cls(img);  // std::expected<std::vector<ClassResult>, Error>
 * @endcode
 */
class OnnxClassifier {
public:
    /**
     * @brief Loads the ONNX model from disk.
     * @param path Path to a `.onnx` model file.
     * @throws improc::ModelError if the file is missing or ORT rejects it.
     */
    explicit OnnxClassifier(const std::filesystem::path& path);

    /**
     * @brief Sets the maximum number of top results to return.
     * @param k Must be positive.
     * @return Reference to this classifier for method chaining.
     * @throws improc::ParameterError if @p k <= 0.
     */
    OnnxClassifier& top_k(int k);

    /**
     * @brief Sets the spatial dimensions the input image is resized to.
     * @param w Width in pixels; must be positive.
     * @param h Height in pixels; must be positive.
     * @return Reference to this classifier for method chaining.
     * @throws improc::ParameterError if either dimension is not positive.
     */
    OnnxClassifier& input_size(int w, int h);

    /**
     * @brief Sets the per-channel mean (B, G, R order) subtracted before inference.
     * @param b Blue channel mean.
     * @param g Green channel mean.
     * @param r Red channel mean.
     * @return Reference to this classifier for method chaining.
     */
    OnnxClassifier& mean(float b, float g, float r);

    /**
     * @brief Sets the pixel value scale factor applied after type conversion.
     * @param s Must be positive; default is `1/255`.
     * @return Reference to this classifier for method chaining.
     * @throws improc::ParameterError if @p s <= 0.
     */
    OnnxClassifier& scale(float s);

    /**
     * @brief Controls BGR→RGB channel swap before inference.
     * @param v `true` swaps channels (required for models trained on RGB data).
     * @return Reference to this classifier for method chaining.
     */
    OnnxClassifier& swap_rb(bool v);

    /**
     * @brief Sets class label strings used to populate `ClassResult::label`.
     * @param lbls Label strings ordered by class index.
     * @return Reference to this classifier for method chaining.
     */
    OnnxClassifier& labels(std::vector<std::string> lbls);

    /**
     * @brief Runs classification inference on the given image.
     * @param img Input image in BGR format.
     * @return Up to `top_k` `ClassResult` entries sorted by score descending,
     *         or an `improc::Error` if inference fails.
     */
    std::expected<std::vector<ClassResult>, improc::Error>
    operator()(const Image<BGR>& img) const;

private:
    OnnxSession              session_;
    int                      top_k_    = 5;
    int                      input_w_  = 224;
    int                      input_h_  = 224;
    float                    mean_b_   = 0.0f;
    float                    mean_g_   = 0.0f;
    float                    mean_r_   = 0.0f;
    float                    scale_    = 1.0f / 255.0f;
    bool                     swap_rb_  = true;
    std::vector<std::string> labels_;
};

} // namespace improc::onnx
