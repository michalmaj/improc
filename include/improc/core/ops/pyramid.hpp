// include/improc/core/ops/pyramid.hpp
#pragma once
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"

namespace improc::core {

/**
 * @brief Pipeline op: reduces image resolution by half (Gaussian pyramid down-step).
 *
 * Applies `cv::pyrDown`. Output dimensions are `ceil(rows/2) × ceil(cols/2)`.
 * Works on any `Image<Format>`.
 *
 * @code
 * Image<Gray> small = img | PyrDown{};
 * @endcode
 */
struct PyrDown {
    template<AnyFormat F>
    Image<F> operator()(Image<F> img) const {
        cv::Mat dst;
        cv::pyrDown(img.mat(), dst);
        return Image<F>(std::move(dst));
    }
};

/**
 * @brief Pipeline op: doubles image resolution (Gaussian pyramid up-step).
 *
 * Applies `cv::pyrUp`. Output dimensions are `2*rows × 2*cols`.
 * Works on any `Image<Format>`.
 *
 * @code
 * Image<Gray> large = img | PyrUp{};
 * @endcode
 */
struct PyrUp {
    template<AnyFormat F>
    Image<F> operator()(Image<F> img) const {
        cv::Mat dst;
        cv::pyrUp(img.mat(), dst);
        return Image<F>(std::move(dst));
    }
};

} // namespace improc::core
