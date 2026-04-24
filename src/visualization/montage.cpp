// src/visualization/montage.cpp
#include "improc/visualization/montage.hpp"
#include <climits>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace improc::visualization {

Montage::Montage(std::vector<Image<BGR>> images) {
    if (images.empty())
        throw ParameterError{"images", "must not be empty", "Montage"};
    images_ = std::move(images);
}

Montage& Montage::cols(int c) {
    if (c <= 0) throw ParameterError{"cols", "must be > 0", "Montage"};
    cols_ = c;
    return *this;
}

Montage& Montage::cell_size(int w, int h) {
    if (w <= 0) throw ParameterError{"cell_width",  "must be > 0", "Montage"};
    if (h <= 0) throw ParameterError{"cell_height", "must be > 0", "Montage"};
    cell_w_ = w;
    cell_h_ = h;
    return *this;
}

Montage& Montage::gap(int g) {
    if (g < 0) throw ParameterError{"gap", "must be >= 0", "Montage"};
    gap_ = g;
    return *this;
}

Montage& Montage::background(cv::Scalar color) {
    bg_ = color;
    return *this;
}

Image<BGR> Montage::operator()() const {
    const int n    = static_cast<int>(images_.size());
    const int cols = cols_ > 0 ? cols_
                               : std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n)))));
    const int rows = static_cast<int>(std::ceil(static_cast<double>(n) / cols));
    const int cw   = cell_w_ > 0 ? cell_w_ : images_[0].cols();
    const int ch   = cell_h_ > 0 ? cell_h_ : images_[0].rows();

    const long long w64 = (long long)cols * cw + (long long)(cols - 1) * gap_;
    const long long h64 = (long long)rows * ch + (long long)(rows - 1) * gap_;
    if (w64 > INT_MAX || h64 > INT_MAX)
        throw ParameterError{"output_dimensions", "total width or height exceeds INT_MAX", "Montage"};
    const int total_w = static_cast<int>(w64);
    const int total_h = static_cast<int>(h64);

    cv::Mat canvas(total_h, total_w, CV_8UC3, bg_);

    for (int i = 0; i < n; i++) {
        const int col = i % cols;
        const int row = i / cols;
        const int x   = col * (cw + gap_);
        const int y   = row * (ch + gap_);

        cv::Mat cell;
        cv::resize(images_[i].mat(), cell, cv::Size(cw, ch));
        cell.copyTo(canvas(cv::Rect(x, y, cw, ch)));
    }

    return Image<BGR>(canvas);
}

} // namespace improc::visualization
