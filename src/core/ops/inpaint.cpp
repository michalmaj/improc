// src/core/ops/inpaint.cpp
#include "improc/core/ops/inpaint.hpp"

namespace improc::core {

Inpaint& Inpaint::radius(double r) {
    radius_ = r;
    return *this;
}

Inpaint& Inpaint::method(InpaintMethod m) {
    method_ = m;
    return *this;
}

Image<BGR> Inpaint::operator()(const Image<BGR>& img, const Image<Gray>& mask) const {
    int flags = (method_ == InpaintMethod::TELEA) ? cv::INPAINT_TELEA : cv::INPAINT_NS;
    cv::Mat result;
    cv::inpaint(img.mat(), mask.mat(), result, radius_, flags);
    return Image<BGR>(std::move(result));
}

} // namespace improc::core
