// src/core/ops/flood_fill.cpp
#include <stdexcept>
#include "improc/core/ops/flood_fill.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

namespace {
void check_seed(int rows, int cols, cv::Point seed) {
    if (seed.x < 0 || seed.y < 0 || seed.x >= cols || seed.y >= rows)
        throw improc::ParameterError{"seed", "is outside image bounds", "FloodFill"};
}
} // namespace

Image<BGR> FloodFill::operator()(const Image<BGR>& img, cv::Point seed, cv::Scalar new_color) const {
    check_seed(img.rows(), img.cols(), seed);
    cv::Mat dst = img.mat().clone();
    cv::floodFill(dst, seed, new_color, nullptr, lo_diff_, up_diff_);
    return Image<BGR>(std::move(dst));
}

Image<Gray> FloodFill::operator()(const Image<Gray>& img, cv::Point seed, uchar new_val) const {
    check_seed(img.rows(), img.cols(), seed);
    cv::Mat dst = img.mat().clone();
    cv::floodFill(dst, seed, cv::Scalar(new_val), nullptr, lo_diff_, up_diff_);
    return Image<Gray>(std::move(dst));
}

} // namespace improc::core
