// include/improc/core/ops/edge.hpp
#pragma once

#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

// Sobel edge detector — returns Image<Gray> containing the gradient magnitude.
//
// Computes Sobel in X and Y (CV_32F), takes the magnitude, and converts back
// to CV_8U with saturate_cast. Works on Image<Gray> directly; BGR input is
// converted to gray first.
//
// ksize must be 1, 3, 5, or 7 (OpenCV requirement). Default: 3.
struct SobelEdge {
    SobelEdge& ksize(int k) {
        if (k != 1 && k != 3 && k != 5 && k != 7)
            throw ParameterError{"ksize", "must be 1, 3, 5, or 7", "SobelEdge"};
        ksize_ = k;
        return *this;
    }

    Image<Gray> operator()(Image<Gray> img) const;
    Image<Gray> operator()(Image<BGR>  img) const;

private:
    int ksize_ = 3;
};

// Canny edge detector — returns Image<Gray> with binary edges.
//
// threshold1 and threshold2 are the hysteresis thresholds for edge linking.
// A good starting point: threshold2 = 3 * threshold1.
// aperture_size is the Sobel kernel size (3, 5, or 7). Default: 3.
//
// Works on Image<Gray> directly; BGR input is converted to gray first.
struct CannyEdge {
    CannyEdge& threshold1(double t) {
        if (t < 0.0)
            throw ParameterError{"threshold1", "must be >= 0", "CannyEdge"};
        threshold1_ = t;
        return *this;
    }
    CannyEdge& threshold2(double t) {
        if (t < 0.0)
            throw ParameterError{"threshold2", "must be >= 0", "CannyEdge"};
        threshold2_ = t;
        return *this;
    }
    CannyEdge& aperture_size(int s) {
        if (s != 3 && s != 5 && s != 7)
            throw ParameterError{"aperture_size", "must be 3, 5, or 7", "CannyEdge"};
        aperture_size_ = s;
        return *this;
    }

    Image<Gray> operator()(Image<Gray> img) const;
    Image<Gray> operator()(Image<BGR>  img) const;

private:
    double threshold1_    = 100.0;
    double threshold2_    = 200.0;
    int    aperture_size_ = 3;
};

} // namespace improc::core
