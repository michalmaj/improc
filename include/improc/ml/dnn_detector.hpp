// include/improc/ml/dnn_detector.hpp
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
 * @brief OpenCV DNN-backed object detector with NMS post-processing.
 *
 * Supports two output formats selectable via `Style`: YOLO-style single-blob
 * output and SSD-style dual-blob (boxes + scores) output. Bounding boxes are
 * rescaled to the original image dimensions and filtered through NMS before
 * being returned as `Detection` values.
 *
 * Supported model formats: `.onnx`, `.pb`, `.caffemodel`, `.weights`, `.t7`, `.net`.
 *
 * @code
 * auto dets = DnnDetector{"yolov8n.onnx"}.confidence_threshold(0.5f)(frame);
 * @endcode
 */
struct DnnDetector {
    /**
     * @brief Selects the output format of the detection model.
     */
    enum class Style {
        YOLO, ///< Single output blob `[1, N, 5+C]`: cx, cy, w, h, obj_conf, class_scores…
        SSD   ///< Two output blobs — boxes `[1,N,4]` (y1,x1,y2,x2 normalised) + scores `[1,N,C]`.
    };

    /**
     * @brief Loads the model from disk.
     * @param model_path Path to the model file.
     * @throws improc::ModelError if the file does not exist, has an unsupported
     *         extension, or OpenCV fails to parse it.
     */
    explicit DnnDetector(std::string model_path);

    /**
     * @brief Sets the output format of the detection model.
     * @param s `Style::YOLO` (default) or `Style::SSD`.
     * @return Reference to this detector for method chaining.
     */
    DnnDetector& style(Style s) { style_ = s; return *this; }

    /**
     * @brief Sets the name of the single output layer (YOLO style).
     * @param n Layer name as reported by the model.
     * @return Reference to this detector for method chaining.
     */
    DnnDetector& output_layer(std::string n) { output_layer_ = std::move(n); return *this; }

    /**
     * @brief Sets the name of the bounding-box output layer (SSD style).
     * @param n Layer name as reported by the model.
     * @return Reference to this detector for method chaining.
     */
    DnnDetector& boxes_layer(std::string n) { boxes_layer_ = std::move(n); return *this; }

    /**
     * @brief Sets the name of the class-scores output layer (SSD style).
     * @param n Layer name as reported by the model.
     * @return Reference to this detector for method chaining.
     */
    DnnDetector& scores_layer(std::string n) { scores_layer_ = std::move(n); return *this; }

    /**
     * @brief Sets the minimum confidence required to keep a detection.
     * @param t Threshold in [0, 1]; default is 0.5.
     * @return Reference to this detector for method chaining.
     * @throws improc::ParameterError if @p t is outside [0, 1].
     */
    DnnDetector& confidence_threshold(float t) {
        if (t < 0.0f || t > 1.0f) throw ParameterError{"confidence_threshold", "must be in [0, 1]", "DnnDetector"};
        confidence_threshold_ = t; return *this;
    }

    /**
     * @brief Sets the IoU threshold used during NMS.
     * @param t Threshold in [0, 1]; default is 0.4.
     * @return Reference to this detector for method chaining.
     * @throws improc::ParameterError if @p t is outside [0, 1].
     */
    DnnDetector& nms_threshold(float t) {
        if (t < 0.0f || t > 1.0f) throw ParameterError{"nms_threshold", "must be in [0, 1]", "DnnDetector"};
        nms_threshold_ = t; return *this;
    }

    /**
     * @brief Sets the spatial dimensions of the input blob.
     * @param w Width in pixels; must be positive.
     * @param h Height in pixels; must be positive.
     * @return Reference to this detector for method chaining.
     * @throws improc::ParameterError if either dimension is not positive.
     */
    DnnDetector& input_size(int w, int h) {
        if (w <= 0 || h <= 0) throw ParameterError{"input_size", "dimensions must be positive", "DnnDetector"};
        input_w_ = w; input_h_ = h; return *this;
    }

    /**
     * @brief Sets the per-channel mean subtracted from the input blob.
     * @param m Mean values as a `cv::Scalar` (B, G, R).
     * @return Reference to this detector for method chaining.
     */
    DnnDetector& mean(cv::Scalar m) { mean_ = m; return *this; }

    /**
     * @brief Sets the pixel value scaling factor applied to the input blob.
     * @param s Scale factor; must be positive (default: 1/255).
     * @return Reference to this detector for method chaining.
     * @throws improc::ParameterError if @p s <= 0.
     */
    DnnDetector& scale(float s) {
        if (s <= 0.0f) throw ParameterError{"scale", "must be positive", "DnnDetector"};
        scale_ = s; return *this;
    }

    /**
     * @brief Controls whether R and B channels are swapped when building the blob.
     * @param s `true` to swap (BGR → RGB); default is `true`.
     * @return Reference to this detector for method chaining.
     */
    DnnDetector& swap_rb(bool s) { swap_rb_ = s; return *this; }

    /**
     * @brief Sets the class label strings used to populate `Detection::label`.
     * @param l Vector of label strings ordered by class index.
     * @return Reference to this detector for method chaining.
     */
    DnnDetector& labels(std::vector<std::string> l) { labels_ = std::move(l); return *this; }

    /**
     * @brief Runs detection inference on the given image.
     *
     * Boxes are rescaled from blob coordinates back to the original image size
     * and filtered through NMS before being returned.
     *
     * @param img Input image in BGR format.
     * @return `Detection` entries that passed confidence and NMS filtering.
     * @throws improc::Exception if the forward pass or output parsing fails.
     */
    std::vector<Detection> operator()(const Image<BGR>& img) const;

private:
    mutable cv::dnn::Net     net_;
    Style                    style_                = Style::YOLO;
    std::string              output_layer_;
    std::string              boxes_layer_;
    std::string              scores_layer_;
    float                    confidence_threshold_ = 0.5f;
    float                    nms_threshold_        = 0.4f;
    int                      input_w_             = 640;
    int                      input_h_             = 640;
    cv::Scalar               mean_                = {0, 0, 0};
    float                    scale_               = 1.0f / 255.0f;
    bool                     swap_rb_             = true;
    std::vector<std::string> labels_;

    /// @brief Parses YOLO-style single output blob into `Detection` entries.
    std::vector<Detection> parse_yolo(const cv::Mat& output, int orig_w, int orig_h) const;
    /// @brief Parses SSD-style dual output blobs into `Detection` entries.
    std::vector<Detection> parse_ssd(const cv::Mat& boxes, const cv::Mat& scores, int orig_w, int orig_h) const;

    /// @brief Returns the set of supported model file extensions.
    static const std::unordered_set<std::string>& valid_extensions();
};

} // namespace improc::ml
