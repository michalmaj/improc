// src/core/ops/connected_components.cpp
#include "improc/core/ops/connected_components.hpp"

namespace improc::core {

void ComponentMap::check(int label) const {
    if (label < 0 || label >= num_labels)
        throw std::out_of_range{"ComponentMap: label out of range"};
}

int ComponentMap::area(int label) const {
    check(label);
    return stats.at<int>(label, cv::CC_STAT_AREA);
}

cv::Rect ComponentMap::bounding_rect(int label) const {
    check(label);
    return {
        stats.at<int>(label, cv::CC_STAT_LEFT),
        stats.at<int>(label, cv::CC_STAT_TOP),
        stats.at<int>(label, cv::CC_STAT_WIDTH),
        stats.at<int>(label, cv::CC_STAT_HEIGHT)
    };
}

cv::Point2d ComponentMap::centroid(int label) const {
    check(label);
    return {centroids.at<double>(label, 0), centroids.at<double>(label, 1)};
}

cv::Mat ComponentMap::mask(int label) const {
    check(label);
    cv::Mat m;
    cv::compare(labels, label, m, cv::CMP_EQ);
    return m;
}

ComponentMap ConnectedComponents::operator()(Image<Gray> img) const {
    ComponentMap result;
    result.num_labels = cv::connectedComponentsWithStats(
        img.mat(), result.labels, result.stats, result.centroids,
        static_cast<int>(conn_), CV_32S);
    return result;
}

} // namespace improc::core
