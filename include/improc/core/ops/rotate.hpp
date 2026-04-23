#pragma once

#include <optional>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Rotates an image around its center by an arbitrary angle.
 *
 * Output size equals input size; pixels that rotate outside the canvas are cropped.
 * `.angle()` must be called before `operator()`.
 *
 * @throws improc::ParameterError if angle is not set.
 * @throws improc::ParameterError if scale <= 0.
 *
 * @code
 * Image<BGR> rotated = img | Rotate{}.angle(45.0);
 * Image<BGR> scaled  = img | Rotate{}.angle(30.0).scale(0.8);
 * @endcode
 */
struct Rotate {
    /// @brief Sets rotation angle in degrees (counter-clockwise positive).
    Rotate& angle(double deg) { angle_ = deg; return *this; }
    /// @brief Sets the uniform scale factor applied during rotation. Default 1.0.
    Rotate& scale(double s) {
        if (s <= 0.0) throw ParameterError{"scale", "must be positive", "Rotate"};
        scale_ = s;
        return *this;
    }

    /// @brief Rotates img around its center by the configured angle.
    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        if (!angle_)
            throw ParameterError{"angle", "must be set before calling operator()", "Rotate"};
        cv::Point2f center(img.cols() / 2.0f, img.rows() / 2.0f);
        cv::Mat M = cv::getRotationMatrix2D(center, *angle_, scale_);
        cv::Mat dst;
        cv::warpAffine(img.mat(), dst, M, cv::Size(img.cols(), img.rows()));
        return Image<Format>(std::move(dst));
    }

private:
    std::optional<double> angle_;
    double scale_ = 1.0;
};

} // namespace improc::core
