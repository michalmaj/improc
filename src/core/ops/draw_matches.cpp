// src/core/ops/draw_matches.cpp
#include "improc/core/ops/draw_matches.hpp"

namespace improc::core {

Image<BGR> DrawKeypoints::operator()(Image<Gray> img) const {
    cv::Mat out;
    cv::drawKeypoints(img.mat(), kps_.keypoints, out,
                      cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return Image<BGR>(out);
}

Image<BGR> DrawKeypoints::operator()(Image<BGR> img) const {
    cv::Mat out;
    cv::drawKeypoints(img.mat(), kps_.keypoints, out,
                      cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return Image<BGR>(out);
}

Image<BGR> DrawMatches::operator()() const {
    cv::Mat out;
    cv::drawMatches(img1_.mat(), kps1_.keypoints,
                    img2_.mat(), kps2_.keypoints,
                    ms_.matches, out);
    return Image<BGR>(out);
}

} // namespace improc::core
