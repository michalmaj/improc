// src/core/ops/distance_transform.cpp
#include "improc/core/ops/distance_transform.hpp"

namespace improc::core {

namespace {

int to_cv_dist_type(DistanceTransform::DistType t) {
    switch (t) {
        case DistanceTransform::DistType::L1: return cv::DIST_L1;
        case DistanceTransform::DistType::L2: return cv::DIST_L2;
        case DistanceTransform::DistType::C:  return cv::DIST_C;
    }
    std::unreachable();
}

int to_cv_mask_size(DistanceTransform::MaskSize m) {
    switch (m) {
        case DistanceTransform::MaskSize::Mask3:   return cv::DIST_MASK_3;
        case DistanceTransform::MaskSize::Mask5:   return cv::DIST_MASK_5;
        case DistanceTransform::MaskSize::Precise: return cv::DIST_MASK_PRECISE;
    }
    std::unreachable();
}

} // namespace

Image<Float32> DistanceTransform::operator()(Image<Gray> img) const {
    cv::Mat dst;
    cv::distanceTransform(img.mat(), dst,
                          to_cv_dist_type(dist_type_),
                          to_cv_mask_size(mask_size_));
    return Image<Float32>(std::move(dst));
}

} // namespace improc::core
