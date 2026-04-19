// src/core/ops/homography.cpp
#include "improc/core/ops/homography.hpp"
#include <opencv2/calib3d.hpp>

namespace improc::core {

std::expected<cv::Mat, improc::Error>
find_homography(const std::vector<cv::Point2f>& src,
                const std::vector<cv::Point2f>& dst,
                double ransac_threshold) {
    if (src.size() < 4 || dst.size() < 4 || src.size() != dst.size())
        return std::unexpected(improc::Error::insufficient_points(
            std::min(src.size(), dst.size())));

    cv::Mat H = cv::findHomography(src, dst, cv::RANSAC, ransac_threshold);
    if (H.empty())
        return std::unexpected(improc::Error::homography_failed());

    return H;
}

} // namespace improc::core
