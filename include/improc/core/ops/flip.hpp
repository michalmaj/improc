#pragma once

#include <stdexcept>
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/core/ops/axis.hpp"

namespace improc::core {

struct Flip {
    explicit Flip(Axis axis) : axis_(axis) {}

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        int flip_code;
        switch (axis_) {
            case Axis::Horizontal: flip_code =  1; break;
            case Axis::Vertical:   flip_code =  0; break;
            case Axis::Both:       flip_code = -1; break;
            default: throw std::invalid_argument("Flip: unknown Axis value");
        }
        cv::Mat dst;
        cv::flip(img.mat(), dst, flip_code);
        return Image<Format>(std::move(dst));
    }

private:
    Axis axis_;
};

} // namespace improc::core
