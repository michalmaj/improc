// src/calib/undistort.cpp
#include "improc/calib/ops/undistort.hpp"

namespace improc::calib {

UndistortMapResult UndistortMap::operator()(cv::Size image_size) const {
    if (K_.empty())
        throw improc::ParameterError{"K", "must be set before calling operator()", "UndistortMap"};
    if (dist_.empty())
        throw improc::ParameterError{"dist", "must be set before calling operator()", "UndistortMap"};
    if (image_size.width <= 0 || image_size.height <= 0)
        throw improc::ParameterError{"image_size", "dimensions must be positive", "UndistortMap"};
    cv::Mat R = R_.empty() ? cv::Mat::eye(3, 3, CV_64F) : R_;
    cv::Mat new_K = new_K_.empty() ? K_ : new_K_;
    UndistortMapResult result;
    cv::initUndistortRectifyMap(K_, dist_, R, new_K,
                                image_size, CV_32FC1,
                                result.map1, result.map2);
    return result;
}

} // namespace improc::calib
