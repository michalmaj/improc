// include/improc/ml/dnn_segmentor.hpp
#pragma once

#include <expected>
#include <filesystem>
#include <string>
#include <vector>
#include <opencv2/dnn.hpp>
#include "improc/core/image.hpp"
#include "improc/ml/result_types.hpp"
#include "improc/exceptions.hpp"
#include "improc/error.hpp"

namespace improc::ml {

using improc::core::Image;
using improc::core::BGR;
using improc::core::Gray;

/**
 * @brief OpenCV DNN-backed semantic segmentation op.
 *
 * Accepts any model that `cv::dnn::readNet` can load (ONNX, Caffe, TF, Darknet)
 * and produces a `[1,C,H,W]` output blob. Argmax over C gives a per-pixel class
 * mask at the original input resolution (via INTER_NEAREST resize).
 *
 * @code
 * auto mask = DnnSegmentor{"deeplabv3.onnx"}.input_size(512, 512)(frame);
 * @endcode
 */
struct DnnSegmentor {
    /**
     * @brief Loads the model from disk.
     * @param model  Path to the model file (ONNX, Caffe, TF, Darknet, etc.).
     * @param config Optional config file path (e.g. `.prototxt` for Caffe).
     * @throws improc::ModelError if the file does not exist or OpenCV fails to parse it.
     */
    explicit DnnSegmentor(const std::filesystem::path& model,
                          const std::filesystem::path& config = "");

    /**
     * @brief Sets the spatial dimensions of the input blob.
     * @param w Width in pixels; must be positive.
     * @param h Height in pixels; must be positive.
     * @return Reference to this segmentor for method chaining.
     * @throws improc::ParameterError if either dimension is not positive.
     */
    DnnSegmentor& input_size(int w, int h) {
        if (w <= 0 || h <= 0)
            throw ParameterError{"input_size", "dimensions must be positive", "DnnSegmentor"};
        input_w_ = w; input_h_ = h; return *this;
    }

    /**
     * @brief Sets the per-channel mean subtracted from the input blob.
     * @param b Blue channel mean.
     * @param g Green channel mean.
     * @param r Red channel mean.
     * @return Reference to this segmentor for method chaining.
     */
    DnnSegmentor& mean(float b, float g, float r) {
        mean_b_ = b; mean_g_ = g; mean_r_ = r; return *this;
    }

    /**
     * @brief Sets the pixel value scaling factor applied to the input blob.
     * @param s Scale factor; must be positive (default: 1/255).
     * @return Reference to this segmentor for method chaining.
     * @throws improc::ParameterError if @p s <= 0.
     */
    DnnSegmentor& scale(float s) {
        if (s <= 0.0f)
            throw ParameterError{"scale", "must be positive", "DnnSegmentor"};
        scale_ = s; return *this;
    }

    /**
     * @brief Controls whether R and B channels are swapped when building the blob.
     * @param v `true` to swap (BGR -> RGB); default is `true`.
     * @return Reference to this segmentor for method chaining.
     */
    DnnSegmentor& swap_rb(bool v) { swap_rb_ = v; return *this; }

    /**
     * @brief Sets the name of the output layer to forward through.
     * @param name Layer name as reported by the model; empty means the last output layer.
     * @return Reference to this segmentor for method chaining.
     */
    DnnSegmentor& output_layer(std::string name) { output_layer_ = std::move(name); return *this; }

    /**
     * @brief Sets the class label strings used to populate `SegmentationMask::labels`.
     * @param lbls Vector of label strings ordered by class index.
     * @return Reference to this segmentor for method chaining.
     */
    DnnSegmentor& labels(std::vector<std::string> lbls) { labels_ = std::move(lbls); return *this; }

    /**
     * @brief Runs segmentation inference on the given image.
     *
     * The network output `[1,C,H,W]` is argmax-reduced over C to produce a
     * per-pixel class mask, then resized (INTER_NEAREST) back to the original
     * input image dimensions.
     *
     * @param img Input image in BGR format.
     * @return `SegmentationMask` on success, or `improc::Error` on inference failure.
     */
    [[nodiscard]] std::expected<SegmentationMask, improc::Error>
    operator()(const Image<BGR>& img) const;

private:
    mutable cv::dnn::Net     net_;
    int                      input_w_       = 512;
    int                      input_h_       = 512;
    float                    mean_b_        = 0.0f;
    float                    mean_g_        = 0.0f;
    float                    mean_r_        = 0.0f;
    float                    scale_         = 1.0f / 255.0f;
    bool                     swap_rb_       = true;
    std::string              output_layer_;
    std::vector<std::string> labels_;
};

} // namespace improc::ml
