// include/improc/ml/result_types.hpp
#pragma once

#include <string>
#include <opencv2/core.hpp>

namespace improc::ml {

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

} // namespace improc::ml
