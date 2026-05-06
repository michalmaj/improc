// include/improc/core/ops/distance_transform.hpp
#pragma once
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

/**
 * @brief Pipeline op: computes distance transform of a binary `Image<Gray>`.
 *
 * Returns `Image<Float32>`. Each pixel is the distance to the nearest zero
 * (background) pixel. Wraps `cv::distanceTransform`.
 *
 * @code
 * Image<Float32> dt = binary | DistanceTransform{};
 * Image<Float32> l1 = binary | DistanceTransform{}.dist_type(DistanceTransform::DistType::L1);
 * @endcode
 */
struct DistanceTransform {
    enum class DistType { L1, L2, C };
    enum class MaskSize { Mask3, Mask5, Precise };

    DistanceTransform& dist_type(DistType t) { dist_type_ = t; return *this; }
    DistanceTransform& mask_size(MaskSize m) { mask_size_ = m; return *this; }

    Image<Float32> operator()(Image<Gray> img) const;

private:
    DistType dist_type_ = DistType::L2;
    MaskSize mask_size_ = MaskSize::Mask3;
};

} // namespace improc::core
