// include/improc/core/ops/pad.hpp
#pragma once

#include <utility>
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

enum class PadMode { Constant, Reflect, Replicate };

namespace detail {
inline int pad_mode_to_cv(PadMode m) {
    switch (m) {
        case PadMode::Constant:  return cv::BORDER_CONSTANT;
        case PadMode::Reflect:   return cv::BORDER_REFLECT;
        case PadMode::Replicate: return cv::BORDER_REPLICATE;
    }
    std::unreachable();
}
} // namespace detail

struct Pad {
    Pad& top(int v) {
        if (v < 0) throw ParameterError{"top", "must be >= 0", "Pad"};
        top_ = v; return *this;
    }
    Pad& bottom(int v) {
        if (v < 0) throw ParameterError{"bottom", "must be >= 0", "Pad"};
        bottom_ = v; return *this;
    }
    Pad& left(int v) {
        if (v < 0) throw ParameterError{"left", "must be >= 0", "Pad"};
        left_ = v; return *this;
    }
    Pad& right(int v) {
        if (v < 0) throw ParameterError{"right", "must be >= 0", "Pad"};
        right_ = v; return *this;
    }
    Pad& mode(PadMode m)      { mode_  = m; return *this; }
    Pad& value(cv::Scalar v)  { value_ = v; return *this; }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        if (top_ == 0 && bottom_ == 0 && left_ == 0 && right_ == 0)
            throw ParameterError{"top/bottom/left/right", "at least one side must be > 0", "Pad"};
        cv::Mat dst;
        try {
            cv::copyMakeBorder(img.mat(), dst,
                               top_, bottom_, left_, right_,
                               detail::pad_mode_to_cv(mode_), value_);
        } catch (const cv::Exception& e) {
            throw ParameterError{"mode", std::string(e.what()), "Pad"};
        }
        return Image<Format>(std::move(dst));
    }

private:
    int        top_ = 0, bottom_ = 0, left_ = 0, right_ = 0;
    PadMode    mode_  = PadMode::Constant;
    cv::Scalar value_ = {0, 0, 0, 0};
};

struct PadToSquare {
    PadToSquare& mode(PadMode m)     { mode_  = m; return *this; }
    PadToSquare& value(cv::Scalar v) { value_ = v; return *this; }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        const int h = img.rows();
        const int w = img.cols();
        if (h == w) return img.clone();

        const int diff   = std::abs(h - w);
        const int pad1   = diff / 2;
        const int pad2   = diff - pad1;
        const int top    = (h < w) ? pad1 : 0;
        const int bottom = (h < w) ? pad2 : 0;
        const int left   = (w < h) ? pad1 : 0;
        const int right  = (w < h) ? pad2 : 0;

        cv::Mat dst;
        try {
            cv::copyMakeBorder(img.mat(), dst,
                               top, bottom, left, right,
                               detail::pad_mode_to_cv(mode_), value_);
        } catch (const cv::Exception& e) {
            throw ParameterError{"mode", std::string(e.what()), "PadToSquare"};
        }
        return Image<Format>(std::move(dst));
    }

private:
    PadMode    mode_  = PadMode::Constant;
    cv::Scalar value_ = {0, 0, 0, 0};
};

} // namespace improc::core
