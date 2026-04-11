#pragma once

#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/ops/axis.hpp"

namespace improc::core {

struct Flip {
    explicit Flip(Axis axis) : axis_(axis) {}

    template<typename Format>
    Image<Format> operator()(Image<Format> img) const {
        const int flip_code = (axis_ == Axis::Horizontal) ?  1 :
                              (axis_ == Axis::Vertical)   ?  0 : -1;
        cv::Mat dst;
        cv::flip(img.mat(), dst, flip_code);
        return Image<Format>(std::move(dst));
    }

private:
    Axis axis_;
};

} // namespace improc::core
