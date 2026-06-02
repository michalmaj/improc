// include/improc/visualization/draw_contours.hpp
#pragma once
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/ops/contours.hpp"
#include "improc/exceptions.hpp"

namespace improc::visualization {

using improc::core::Image;
using improc::core::BGR;
using improc::core::ContourSet;

/**
 * @brief Pipeline op: draws contours from a `ContourSet` onto a BGR image clone.
 *
 * Wraps `cv::drawContours`. Pass `index(-1)` to draw all (default).
 * Pass `thickness(-1)` to fill contours.
 * The source image is never mutated.
 *
 * @throws improc::ParameterError if `thickness` is not positive and not -1.
 *
 * @code
 * Image<BGR> annotated = bgr | DrawContours{cs}.color({0, 255, 0});
 * Image<BGR> filled    = bgr | DrawContours{cs}.thickness(-1);
 * @endcode
 */
struct DrawContours {
    explicit DrawContours(ContourSet cs) : cs_(std::move(cs)) {}

    DrawContours& index(int i) { index_ = i; return *this; }

    DrawContours& color(cv::Scalar c) { color_ = c; return *this; }

    DrawContours& thickness(int t) {
        if (t <= 0 && t != -1)
            throw improc::ParameterError{"thickness", "must be positive or -1 (fill)", "DrawContours"};
        thickness_ = t;
        return *this;
    }

    [[nodiscard]] Image<BGR> operator()(Image<BGR> img) const;

private:
    ContourSet cs_;
    int        index_{-1};
    cv::Scalar color_{0, 255, 0};
    int        thickness_{1};
};

} // namespace improc::visualization
