// include/improc/core/ops/haar_detect.hpp
#pragma once
#include <vector>
#include <opencv2/objdetect.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Pipeline functor: detects objects in `Image<BGR>` using a pre-loaded
 *        `cv::CascadeClassifier`. Returns bounding boxes in pixel coordinates.
 *
 * Converts the image to Gray internally. Returns an empty vector when no
 * objects are found. Does not throw during inference.
 *
 * @code
 * cv::CascadeClassifier cc;
 * cc.load("haarcascade_frontalface_default.xml");
 * std::vector<cv::Rect> faces = DetectHaar{}(img, cc);
 * @endcode
 */
struct DetectHaar {
    /// @brief Scale factor between pyramid levels. Must be > 1.0; default 1.1.
    DetectHaar& scale_factor(double f) {
        if (f <= 1.0) throw ParameterError{"scale_factor", "must be > 1.0", "DetectHaar"};
        scale_factor_ = f; return *this;
    }

    /// @brief Minimum number of neighbouring rectangles to retain a detection. Default 3.
    DetectHaar& min_neighbors(int n) {
        if (n < 0) throw ParameterError{"min_neighbors", "must be >= 0", "DetectHaar"};
        min_neighbors_ = n; return *this;
    }

    /// @brief Minimum object size. Default (0,0) = no minimum.
    DetectHaar& min_size(cv::Size s) { min_size_ = s; return *this; }

    /// @brief Maximum object size. Default (0,0) = no maximum.
    DetectHaar& max_size(cv::Size s) { max_size_ = s; return *this; }

    [[nodiscard]] std::vector<cv::Rect>
    operator()(const Image<BGR>& img, const cv::CascadeClassifier& cc) const;

private:
    double   scale_factor_  = 1.1;
    int      min_neighbors_ = 3;
    cv::Size min_size_      = {};
    cv::Size max_size_      = {};
};

} // namespace improc::core
