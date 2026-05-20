// include/improc/core/ops/inpaint.hpp
#pragma once

#include <opencv2/photo.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

enum class InpaintMethod { TELEA, NS };

struct Inpaint {
    Inpaint& radius(double r);
    Inpaint& method(InpaintMethod m);

    Image<BGR> operator()(const Image<BGR>& img, const Image<Gray>& mask) const;

private:
    double radius_ = 3.0;
    InpaintMethod method_ = InpaintMethod::TELEA;
};

} // namespace improc::core
