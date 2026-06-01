// src/calib/chessboard.cpp
#include "improc/calib/ops/chessboard.hpp"
#include "improc/exceptions.hpp"
#include <opencv2/imgproc.hpp>

namespace improc::calib {

FindChessboardResult FindChessboardCorners::operator()(Image<Gray> img) const {
    if (!has_board_size_)
        throw improc::ParameterError{"board_size", "must be set before calling operator()", "FindChessboardCorners"};
    FindChessboardResult result;
    result.found = cv::findChessboardCorners(img.mat(), board_size_,
                                             result.corners, flags_);
    if (!result.found) result.corners.clear();
    return result;
}

FindChessboardResult FindChessboardCorners::operator()(Image<BGR> img) const {
    cv::Mat gray;
    cv::cvtColor(img.mat(), gray, cv::COLOR_BGR2GRAY);
    return (*this)(Image<Gray>(std::move(gray)));
}

FindChessboardResult FindChessboardCornersSB::operator()(Image<Gray> img) const {
    if (!has_board_size_)
        throw improc::ParameterError{"board_size", "must be set before calling operator()", "FindChessboardCornersSB"};
    FindChessboardResult result;
    result.found = cv::findChessboardCornersSB(img.mat(), board_size_,
                                               result.corners, flags_);
    if (!result.found) result.corners.clear();
    return result;
}

FindChessboardResult FindChessboardCornersSB::operator()(Image<BGR> img) const {
    cv::Mat gray;
    cv::cvtColor(img.mat(), gray, cv::COLOR_BGR2GRAY);
    return (*this)(Image<Gray>(std::move(gray)));
}

std::vector<cv::Point2f> RefineCorners::operator()(Image<Gray>              img,
                                                    std::vector<cv::Point2f> corners) const {
    cv::cornerSubPix(img.mat(), corners, win_size_, zero_zone_,
                     cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
                                      max_iter_, epsilon_));
    return corners;
}

} // namespace improc::calib
