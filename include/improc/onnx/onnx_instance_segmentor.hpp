// include/improc/onnx/onnx_instance_segmentor.hpp
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
using improc::ml::SegmentInstance;

/**
 * @brief ONNX Runtime-backed instance segmentor with YOLOv8-seg post-processing.
 *
 * Wraps `OnnxSession` with a full image-to-`SegmentInstance` pipeline:
 * resize → float conversion → mean subtraction → optional channel swap →
 * HWC→CHW → inference → YOLO box parsing → NMS → prototype mask decoding.
 *
 * Expects two output tensors in YOLOv8-seg format:
 * - Output 0: `[1, 4+C+32, N]` — boxes (cx,cy,w,h) + class scores + mask coefficients.
 * - Output 1: `[1, 32, Hm, Wm]` — prototype masks.
 *
 * @code
 * auto seg = OnnxInstanceSegmentor{"yolov8n-seg.onnx"}.confidence_threshold(0.5f);
 * auto instances = seg(img);  // std::expected<std::vector<SegmentInstance>, Error>
 * @endcode
 */
class OnnxInstanceSegmentor {
public:
    /**
     * @brief Loads the ONNX model from disk.
     * @param path Path to a `.onnx` model file.
     * @throws improc::ModelError if the file is missing or ORT rejects it.
     */
    explicit OnnxInstanceSegmentor(const std::filesystem::path& path);

    /**
     * @brief Sets the spatial dimensions the input image is resized to.
     * @throws improc::ParameterError if either dimension is not positive.
     */
    OnnxInstanceSegmentor& input_size(int w, int h);

    /** @brief Sets the per-channel mean (B, G, R order) subtracted before inference. */
    OnnxInstanceSegmentor& mean(float b, float g, float r);

    /**
     * @brief Sets the pixel value scale factor applied after type conversion.
     * @throws improc::ParameterError if @p s <= 0.
     */
    OnnxInstanceSegmentor& scale(float s);

    /** @brief Controls BGR→RGB channel swap before inference. */
    OnnxInstanceSegmentor& swap_rb(bool v);

    /** @brief Sets class label strings used to populate `SegmentInstance::label`. */
    OnnxInstanceSegmentor& labels(std::vector<std::string> lbls);

    /**
     * @brief Sets the minimum class score required to keep a detection.
     * @param t In [0, 1]; default 0.5.
     * @throws improc::ParameterError if outside [0, 1].
     */
    OnnxInstanceSegmentor& confidence_threshold(float t);

    /**
     * @brief Sets the IoU threshold used during NMS.
     * @param t In [0, 1]; default 0.4.
     * @throws improc::ParameterError if outside [0, 1].
     */
    OnnxInstanceSegmentor& nms_threshold(float t);

    /**
     * @brief Sets the sigmoid threshold for binarising decoded prototype masks.
     * @param t In [0, 1]; default 0.5.
     * @throws improc::ParameterError if outside [0, 1].
     */
    OnnxInstanceSegmentor& mask_threshold(float t);

    /**
     * @brief Runs instance segmentation inference on the given image.
     * @param img Input image in BGR format.
     * @return Instances that passed confidence and NMS filtering,
     *         each with a binary mask at the original image resolution,
     *         or an `improc::Error` if inference fails.
     */
    [[nodiscard]] std::expected<std::vector<SegmentInstance>, improc::Error>
    operator()(const Image<BGR>& img) const;

private:
    OnnxSession              session_;
    float                    confidence_threshold_ = 0.5f;
    float                    nms_threshold_        = 0.4f;
    float                    mask_threshold_       = 0.5f;
    int                      input_w_              = 640;
    int                      input_h_              = 640;
    float                    mean_b_               = 0.0f;
    float                    mean_g_               = 0.0f;
    float                    mean_r_               = 0.0f;
    float                    scale_                = 1.0f / 255.0f;
    bool                     swap_rb_              = true;
    std::vector<std::string> labels_;
};

} // namespace improc::onnx
