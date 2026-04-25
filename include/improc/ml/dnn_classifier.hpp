// include/improc/ml/dnn_classifier.hpp
#pragma once

#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>
#include <opencv2/dnn.hpp>
#include "improc/core/image.hpp"
#include "improc/ml/result_types.hpp"
#include "improc/exceptions.hpp"

namespace improc::ml {

using improc::core::Image;
using improc::core::BGR;

/**
 * @brief OpenCV DNN-backed image classifier.
 *
 * Loads a model at construction, then classifies `Image<BGR>` inputs via
 * `operator()`. Results are sorted by score in descending order and trimmed
 * to the configured @ref top_k count. Fluent setters allow pre- and
 * post-processing parameters to be chained before the first inference call.
 *
 * Supported model formats: `.onnx`, `.pb`, `.caffemodel`, `.weights`, `.t7`, `.net`.
 *
 * @code
 * auto results = img | Resize{}.width(224) | DnnClassifier{"resnet50.onnx"}.top_k(5);
 * @endcode
 */
struct DnnClassifier {
    /**
     * @brief Loads the model from disk.
     * @param model_path Path to the model file.
     * @throws improc::ModelError if the file does not exist, has an unsupported
     *         extension, or OpenCV fails to parse it.
     */
    explicit DnnClassifier(std::string model_path);

    /**
     * @brief Sets the maximum number of top results to return.
     * @param k Number of results; must be positive.
     * @return Reference to this classifier for method chaining.
     * @throws improc::ParameterError if @p k <= 0.
     */
    DnnClassifier& top_k(int k) {
        if (k <= 0) throw ParameterError{"top_k", "must be positive", "DnnClassifier"};
        top_k_ = k; return *this;
    }

    /**
     * @brief Sets the spatial dimensions of the input blob.
     * @param w Width in pixels; must be positive.
     * @param h Height in pixels; must be positive.
     * @return Reference to this classifier for method chaining.
     * @throws improc::ParameterError if either dimension is not positive.
     */
    DnnClassifier& input_size(int w, int h) {
        if (w <= 0 || h <= 0) throw ParameterError{"input_size", "dimensions must be positive", "DnnClassifier"};
        input_w_ = w; input_h_ = h; return *this;
    }

    /**
     * @brief Sets the per-channel mean subtracted from the input blob.
     * @param m Mean values as a `cv::Scalar` (B, G, R).
     * @return Reference to this classifier for method chaining.
     */
    DnnClassifier& mean(cv::Scalar m)   { mean_ = m; return *this; }

    /**
     * @brief Sets the pixel value scaling factor applied to the input blob.
     * @param s Scale factor; must be positive (default: 1/255).
     * @return Reference to this classifier for method chaining.
     * @throws improc::ParameterError if @p s <= 0.
     */
    DnnClassifier& scale(float s) {
        if (s <= 0.0f) throw ParameterError{"scale", "must be positive", "DnnClassifier"};
        scale_ = s; return *this;
    }

    /**
     * @brief Controls whether R and B channels are swapped when building the blob.
     * @param s `true` to swap (BGR → RGB); default is `true`.
     * @return Reference to this classifier for method chaining.
     */
    DnnClassifier& swap_rb(bool s) { swap_rb_ = s; return *this; }

    /**
     * @brief Sets the class label strings used to populate `ClassResult::label`.
     * @param l Vector of label strings ordered by class index.
     * @return Reference to this classifier for method chaining.
     */
    DnnClassifier& labels(std::vector<std::string> l) { labels_ = std::move(l); return *this; }

    /**
     * @brief Runs classification inference on the given image.
     * @param img Input image in BGR format.
     * @return Up to `top_k` `ClassResult` entries sorted by score descending.
     */
    std::vector<ClassResult> operator()(const Image<BGR>& img) const;

private:
    mutable cv::dnn::Net     net_;
    int                      top_k_   = 5;
    int                      input_w_ = 224;
    int                      input_h_ = 224;
    cv::Scalar               mean_    = {0, 0, 0};
    float                    scale_   = 1.0f / 255.0f;
    bool                     swap_rb_ = true;
    std::vector<std::string> labels_;

    /// @brief Returns the set of supported model file extensions.
    static const std::unordered_set<std::string>& valid_extensions();
};

} // namespace improc::ml
