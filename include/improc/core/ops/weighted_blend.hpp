// include/improc/core/ops/weighted_blend.hpp
#pragma once

#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Weighted blend of two images: `output = img1 * alpha + img2 * (1 - alpha)`.
 *
 * Both images must have identical size and format.
 * Default alpha is 0.5 (equal blend).
 *
 * @tparam F  Format tag of both images.
 * @throws ParameterError if alpha is outside [0, 1].
 * @throws ParameterError if image sizes differ.
 *
 * @code
 * Image<BGR> blended = img1 | WeightedBlend<BGR>{img2}.alpha(0.7);
 * @endcode
 */
template<AnyFormat F>
struct WeightedBlend {
    explicit WeightedBlend(Image<F> other) : other_(std::move(other)) {}

    WeightedBlend& alpha(double a) {
        if (a < 0.0 || a > 1.0)
            throw ParameterError{"alpha", "must be in [0, 1]", "WeightedBlend"};
        alpha_ = a;
        return *this;
    }

    Image<F> operator()(Image<F> img) const {
        if (img.rows() != other_.rows() || img.cols() != other_.cols())
            throw ParameterError{"other", "size must match source image", "WeightedBlend"};
        cv::Mat dst;
        cv::addWeighted(img.mat(), alpha_, other_.mat(), 1.0 - alpha_, 0.0, dst);
        return Image<F>(std::move(dst));
    }

private:
    Image<F> other_;
    double   alpha_{0.5};
};

} // namespace improc::core
