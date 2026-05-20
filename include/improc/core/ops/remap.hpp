// include/improc/core/ops/remap.hpp
#pragma once
#include <stdexcept>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"

namespace improc::core {

/**
 * @brief Applies a generic remap (per-pixel displacement) to an image.
 *
 * map1 holds the x-coordinates (column index) of the source pixels;
 * map2 holds the y-coordinates (row index). Both must be CV_32FC1 and
 * have identical sizes.
 *
 * @throws std::invalid_argument if either map is empty or their sizes differ.
 *
 * @code
 * cv::Mat map1, map2;
 * cv::initUndistortRectifyMap(..., map1, map2);
 * Image<BGR> undistorted = img | Remap(map1, map2);
 * @endcode
 */
struct Remap {
    Remap(cv::Mat map1, cv::Mat map2)
        : map1_(std::move(map1)), map2_(std::move(map2)) {
        if (map1_.empty() || map2_.empty())
            throw std::invalid_argument("Remap: maps must not be empty");
        if (map1_.size() != map2_.size())
            throw std::invalid_argument("Remap: map1 and map2 must have the same size");
    }

    /// @brief Sets the interpolation method (default: cv::INTER_LINEAR).
    Remap& interpolation(int flags) { interpolation_ = flags; return *this; }

    template<AnyFormat F>
    Image<F> operator()(Image<F> img) const {
        cv::Mat result;
        cv::remap(img.mat(), result, map1_, map2_, interpolation_);
        return Image<F>(std::move(result));
    }

private:
    cv::Mat map1_;
    cv::Mat map2_;
    int     interpolation_ = cv::INTER_LINEAR;
};

} // namespace improc::core
