// include/improc/onnx/onnx_segmentor.hpp
#pragma once

#include <expected>
#include <filesystem>
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
using improc::ml::SegmentationMask;

/**
 * @brief ONNX Runtime-backed semantic image segmentor.
 *
 * Wraps `OnnxSession` with a full image-to-`SegmentationMask` pipeline:
 * resize → float conversion → mean subtraction → optional channel swap →
 * HWC→CHW → inference → argmax over class axis → resize to original resolution.
 *
 * Expects the model's first output to be a float32 logit map of shape
 * `[1, C, H, W]`. The returned `SegmentationMask::class_mask` is a Gray image
 * at the original input resolution where each pixel value is a class index in [0, C-1].
 *
 * @code
 * auto seg = OnnxSegmentor{"fcn.onnx"}.input_size(512, 512).labels(class_names);
 * auto result = seg(img);  // std::expected<SegmentationMask, Error>
 * @endcode
 */
class OnnxSegmentor {
public:
    /**
     * @brief Loads the ONNX model from disk.
     * @param path Path to a `.onnx` model file.
     * @throws improc::ModelError if the file is missing or ORT rejects it.
     */
    explicit OnnxSegmentor(const std::filesystem::path& path);

    /**
     * @brief Sets the spatial dimensions the input image is resized to.
     * @param w Width in pixels; must be positive.
     * @param h Height in pixels; must be positive.
     * @throws improc::ParameterError if either dimension is not positive.
     */
    OnnxSegmentor& input_size(int w, int h);

    /**
     * @brief Sets the per-channel mean (B, G, R order) subtracted before inference.
     */
    OnnxSegmentor& mean(float b, float g, float r);

    /**
     * @brief Sets the pixel value scale factor applied after type conversion.
     * @param s Must be positive; default is `1/255`.
     * @throws improc::ParameterError if @p s <= 0.
     */
    OnnxSegmentor& scale(float s);

    /**
     * @brief Controls BGR→RGB channel swap before inference.
     * @param v `true` swaps channels (required for models trained on RGB data).
     */
    OnnxSegmentor& swap_rb(bool v);

    /**
     * @brief Sets class label strings used to populate `SegmentationMask::labels`.
     * @param lbls Label strings ordered by class index.
     */
    OnnxSegmentor& labels(std::vector<std::string> lbls);

    /**
     * @brief Runs semantic segmentation inference on the given image.
     *
     * The output mask is resized back to the original input image dimensions
     * using nearest-neighbour interpolation to preserve class IDs.
     *
     * @param img Input image in BGR format.
     * @return `SegmentationMask` with `class_mask` at original resolution,
     *         or an `improc::Error` if inference fails.
     */
    [[nodiscard]] std::expected<SegmentationMask, improc::Error>
    operator()(const Image<BGR>& img) const;

private:
    OnnxSession              session_;
    int                      input_w_  = 512;
    int                      input_h_  = 512;
    float                    mean_b_   = 0.0f;
    float                    mean_g_   = 0.0f;
    float                    mean_r_   = 0.0f;
    float                    scale_    = 1.0f / 255.0f;
    bool                     swap_rb_  = true;
    std::vector<std::string> labels_;
};

} // namespace improc::onnx
