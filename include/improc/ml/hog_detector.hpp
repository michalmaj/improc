// include/improc/ml/hog_detector.hpp
#pragma once

#include <vector>
#include <opencv2/objdetect.hpp>
#include "improc/core/image.hpp"
#include "improc/ml/result_types.hpp"
#include "improc/exceptions.hpp"

namespace improc::ml {

using improc::core::Image;
using improc::core::BGR;

/**
 * @brief People detector using `cv::HOGDescriptor` with the default SVM.
 *
 * Returns `std::vector<Detection>` with `class_id=0`, `label="person"`, and
 * `confidence` computed via sigmoid of the raw HOG weight.
 *
 * @code
 * HOGDetector det;
 * auto people = det(frame);
 * @endcode
 */
struct HOGDetector {
    /// @brief Construct with the default OpenCV people detector SVM.
    HOGDetector();

    /// @brief Construct with a custom SVM descriptor vector. Must be non-empty.
    explicit HOGDetector(const std::vector<float>& svm);

    /// @brief Sliding window stride. Default (8,8).
    HOGDetector& win_stride(cv::Size s)  { win_stride_    = s; return *this; }

    /// @brief Detection window padding. Default (4,4).
    HOGDetector& padding(cv::Size s)     { padding_       = s; return *this; }

    /// @brief Scale step between pyramid levels. Must be > 1.0; default 1.05.
    HOGDetector& scale(double s) {
        if (s <= 1.0)
            throw ParameterError{"scale", "must be > 1.0", "HOGDetector"};
        scale_ = s; return *this;
    }

    /// @brief Classifier hit threshold. Higher = stricter. Default 0.0.
    HOGDetector& hit_threshold(double t) { hit_threshold_ = t; return *this; }

    [[nodiscard]] std::vector<Detection> operator()(const Image<BGR>& img) const;

private:
    cv::HOGDescriptor hog_;
    cv::Size win_stride_    = {8, 8};
    cv::Size padding_       = {4, 4};
    double   scale_         = 1.05;
    double   hit_threshold_ = 0.0;
};

} // namespace improc::ml
