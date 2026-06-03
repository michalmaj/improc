// include/improc/ml/result_types.hpp
#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"

namespace improc::ml {

using improc::core::Image;
using improc::core::Gray;

/**
 * @brief Result of a single classification inference.
 *
 * Returned as one element in the vector produced by `DnnClassifier::operator()`.
 * Results are sorted by @ref score in descending order.
 */
struct ClassResult {
    int         class_id;  ///< Index into the labels vector (or raw softmax output index).
    float       score;     ///< Confidence score — softmax probability or raw logit.
    std::string label;     ///< Human-readable class name; empty if no labels were provided.
};

/**
 * @brief Result of a single object detection.
 *
 * Returned as one element in the vector produced by `DnnDetector::operator()`.
 * Bounding box coordinates are in pixel space on the **original** input image,
 * after NMS filtering.
 */
struct Detection {
    cv::Rect2f  box;        ///< Bounding box in pixel coordinates on the original input image.
    int         class_id;   ///< Index into the labels vector (or raw class index).
    float       confidence; ///< Detection confidence after NMS, in [0, 1].
    std::string label;      ///< Human-readable class name; empty if no labels were provided.
};

/**
 * @brief Result of a semantic segmentation inference.
 *
 * `class_mask` is a Gray image whose pixel values are class indices in [0, C-1],
 * at the original input image resolution. `labels` maps index to name; may be empty.
 */
struct SegmentationMask {
    Image<Gray>              class_mask; ///< Pixel value = class_id; original input resolution.
    std::vector<std::string> labels;     ///< Class names ordered by class_id; may be empty.

    /**
     * @brief Returns the label for the given class id.
     * @return `labels[class_id]` if in range, an empty `string_view` otherwise.
     */
    [[nodiscard]] std::string_view label_at(int class_id) const noexcept {
        if (class_id < 0 || static_cast<std::size_t>(class_id) >= labels.size())
            return {};
        return labels[static_cast<std::size_t>(class_id)];
    }
};

/**
 * @brief One detected instance from `OnnxInstanceSegmentor`.
 *
 * Combines detection metadata (box, class, confidence, label) with a per-instance
 * binary segmentation mask at the original input image resolution.
 */
struct SegmentInstance {
    cv::Rect2f  box;         ///< Bounding box in pixel coordinates on the original input image.
    int         class_id;    ///< Index into the labels vector (or raw class index).
    float       confidence;  ///< Detection confidence after NMS, in [0, 1].
    std::string label;       ///< Human-readable class name; empty if no labels were provided.
    Image<Gray> mask;        ///< Binary mask (0 or 255), full original image size.
};

} // namespace improc::ml
