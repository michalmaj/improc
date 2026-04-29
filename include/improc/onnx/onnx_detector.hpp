// include/improc/onnx/onnx_detector.hpp
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
using improc::ml::Detection;

/**
 * @brief ONNX Runtime-backed object detector with NMS post-processing.
 *
 * Wraps `OnnxSession` with a full image-to-`Detection` pipeline:
 * resize → float conversion → mean subtraction → optional channel swap →
 * HWC→CHW → inference → format-specific parsing → NMS → coordinate rescaling.
 *
 * Two output formats are supported via `Style`:
 * - **YOLO**: single blob `[1, 4+C, N]` — `cx,cy,w,h` + per-class scores.
 * - **SSD**: two blobs — boxes `[1, N, 4]` (y1,x1,y2,x2 normalised) and
 *   scores `[1, N, C]`.
 *
 * Fluent setters mirror the `DnnDetector` API for consistency.
 *
 * @code
 * auto det = OnnxDetector{"yolov8n.onnx"}.confidence_threshold(0.5f);
 * auto boxes = det(frame);  // std::expected<std::vector<Detection>, Error>
 * @endcode
 */
class OnnxDetector {
public:
    /**
     * @brief Selects the output format of the detection model.
     */
    enum class Style {
        YOLO, ///< Single output blob `[1, 4+C, N]`: cx, cy, w, h, class_scores…
        SSD   ///< Two output blobs — boxes `[1,N,4]` (normalised) + scores `[1,N,C]`.
    };

    /**
     * @brief Loads the ONNX model from disk.
     * @param path Path to a `.onnx` model file.
     * @throws improc::ModelError if the file is missing or ORT rejects it.
     */
    explicit OnnxDetector(const std::filesystem::path& path);

    /**
     * @brief Sets the output format of the detection model.
     * @param s `Style::YOLO` (default) or `Style::SSD`.
     * @return Reference to this detector for method chaining.
     */
    OnnxDetector& style(Style s);

    /**
     * @brief Sets the minimum confidence required to keep a detection.
     * @param t Threshold in [0, 1]; default is 0.5.
     * @return Reference to this detector for method chaining.
     * @throws improc::ParameterError if @p t is outside [0, 1].
     */
    OnnxDetector& confidence_threshold(float t);

    /**
     * @brief Sets the IoU threshold used during NMS.
     * @param t Threshold in [0, 1]; default is 0.4.
     * @return Reference to this detector for method chaining.
     * @throws improc::ParameterError if @p t is outside [0, 1].
     */
    OnnxDetector& nms_threshold(float t);

    /**
     * @brief Sets the spatial dimensions the input image is resized to.
     * @param w Width in pixels; must be positive.
     * @param h Height in pixels; must be positive.
     * @return Reference to this detector for method chaining.
     * @throws improc::ParameterError if either dimension is not positive.
     */
    OnnxDetector& input_size(int w, int h);

    /**
     * @brief Sets the per-channel mean (B, G, R order) subtracted before inference.
     * @param b Blue channel mean.
     * @param g Green channel mean.
     * @param r Red channel mean.
     * @return Reference to this detector for method chaining.
     */
    OnnxDetector& mean(float b, float g, float r);

    /**
     * @brief Sets the pixel value scale factor applied after type conversion.
     * @param s Must be positive; default is `1/255`.
     * @return Reference to this detector for method chaining.
     * @throws improc::ParameterError if @p s <= 0.
     */
    OnnxDetector& scale(float s);

    /**
     * @brief Controls BGR→RGB channel swap before inference.
     * @param v `true` swaps channels (required for models trained on RGB data).
     * @return Reference to this detector for method chaining.
     */
    OnnxDetector& swap_rb(bool v);

    /**
     * @brief Sets class label strings used to populate `Detection::label`.
     * @param lbls Label strings ordered by class index.
     * @return Reference to this detector for method chaining.
     */
    OnnxDetector& labels(std::vector<std::string> lbls);

    /**
     * @brief Runs detection inference on the given image.
     *
     * Bounding boxes are rescaled from blob coordinates back to the original
     * image dimensions and filtered through NMS before being returned.
     *
     * @param img Input image in BGR format.
     * @return `Detection` entries that passed confidence and NMS filtering,
     *         or an `improc::Error` if inference fails.
     */
    std::expected<std::vector<Detection>, improc::Error>
    operator()(const Image<BGR>& img) const;

private:
    OnnxSession              session_;
    Style                    style_               = Style::YOLO;
    float                    confidence_threshold_ = 0.5f;
    float                    nms_threshold_        = 0.4f;
    int                      input_w_              = 640;
    int                      input_h_              = 640;
    float                    mean_b_               = 0.0f;
    float                    mean_g_               = 0.0f;
    float                    mean_r_               = 0.0f;
    float                    scale_                = 1.0f / 255.0f;
    bool                     swap_rb_              = true;
    std::vector<std::string> labels_;

    /// @brief Parses a YOLO-style single output tensor into `Detection` entries.
    std::vector<Detection> parse_yolo(const TensorInfo& output,
                                      int orig_w, int orig_h) const;

    /// @brief Parses SSD-style boxes + scores tensors into `Detection` entries.
    std::vector<Detection> parse_ssd(const TensorInfo& boxes,
                                     const TensorInfo& scores,
                                     int orig_w, int orig_h) const;
};

} // namespace improc::onnx
