// src/core/ops/stitching.cpp
#include "improc/core/ops/stitching.hpp"

namespace improc::core {

StitchResult Stitch::operator()(const std::vector<Image<BGR>>& images) const {
    if (images.size() < 2)
        throw improc::ParameterError{"images",
            "at least 2 images required", "Stitch"};

    auto cv_mode = (mode_ == Mode::Panorama)
        ? cv::Stitcher::PANORAMA
        : cv::Stitcher::SCANS;

    std::vector<cv::Mat> mats;
    mats.reserve(images.size());
    for (const auto& img : images) mats.push_back(img.mat());

    cv::Mat panorama;
    auto status = cv::Stitcher::create(cv_mode)->stitch(mats, panorama);
    if (status == cv::Stitcher::OK)
        return {true, Image<BGR>(std::move(panorama))};
    return {false, Image<BGR>(cv::Mat(1, 1, CV_8UC3, cv::Scalar(0)))};
}

} // namespace improc::core
