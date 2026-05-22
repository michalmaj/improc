// src/calib/chessboard.cpp
#include "improc/calib/ops/chessboard.hpp"
#include <opencv2/imgproc.hpp>

namespace improc::calib {

FindChessboardResult FindChessboardCorners::operator()(const Image<Gray>& img) const {
    if (!has_board_size_)
        throw std::invalid_argument("FindChessboardCorners: board_size must be set");
    FindChessboardResult result;
    result.found = cv::findChessboardCorners(img.mat(), board_size_,
                                             result.corners, flags_);
    if (!result.found) result.corners.clear();
    return result;
}

FindChessboardResult FindChessboardCorners::operator()(const Image<BGR>& img) const {
    cv::Mat gray;
    cv::cvtColor(img.mat(), gray, cv::COLOR_BGR2GRAY);
    return (*this)(Image<Gray>(std::move(gray)));
}

} // namespace improc::calib
