#pragma once
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

// All reference-based ops throw ParameterError when ref.size() != cmp.size().

struct PSNR {
    // Returns INFINITY when images are identical (MSE == 0).
    double operator()(const Image<BGR>&  ref, const Image<BGR>&  cmp) const;
    double operator()(const Image<Gray>& ref, const Image<Gray>& cmp) const;
};

struct SSIM {
    // Returns 1.0 for identical images.
    double operator()(const Image<BGR>&  ref, const Image<BGR>&  cmp) const;
    double operator()(const Image<Gray>& ref, const Image<Gray>& cmp) const;
};

struct GMSD {
    // Lower = better quality. Returns 0.0 for identical images.
    double operator()(const Image<BGR>&  ref, const Image<BGR>&  cmp) const;
    double operator()(const Image<Gray>& ref, const Image<Gray>& cmp) const;
};

struct MSE {
    // Returns 0.0 for identical images.
    double operator()(const Image<BGR>&  ref, const Image<BGR>&  cmp) const;
    double operator()(const Image<Gray>& ref, const Image<Gray>& cmp) const;
};

} // namespace improc::core
