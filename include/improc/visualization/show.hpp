// include/improc/visualization/show.hpp
#pragma once

#include <string>
#include <opencv2/highgui.hpp>
#include "improc/core/image.hpp"

namespace improc::visualization {

using improc::core::Image;
using improc::core::BGR;

struct Show {
    explicit Show(std::string window_name)
        : window_name_(std::move(window_name)) {}

    Show& wait_ms(int ms) { wait_ms_ = ms; return *this; }

    // Calls cv::imshow + cv::waitKey, then returns img unchanged.
    // wait_ms=0 blocks until a key is pressed (default).
    // wait_ms=1 is suitable for camera loops.
    Image<BGR> operator()(Image<BGR> img) const {
        cv::imshow(window_name_, img.mat());
        cv::waitKey(wait_ms_);
        return img;
    }

private:
    std::string window_name_;
    int         wait_ms_ = 0;
};

} // namespace improc::visualization
