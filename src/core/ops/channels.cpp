// src/core/ops/channels.cpp
#include "improc/core/ops/channels.hpp"

namespace improc::core {

std::array<Image<Gray>, 3> SplitChannels::operator()(const Image<BGR>& img) const {
    std::vector<cv::Mat> planes;
    cv::split(img.mat(), planes);
    return {Image<Gray>(planes[0]), Image<Gray>(planes[1]), Image<Gray>(planes[2])};
}

std::array<Image<Gray>, 4> SplitChannels::operator()(const Image<BGRA>& img) const {
    std::vector<cv::Mat> planes;
    cv::split(img.mat(), planes);
    return {Image<Gray>(planes[0]), Image<Gray>(planes[1]),
            Image<Gray>(planes[2]), Image<Gray>(planes[3])};
}

static void check_sizes(const Image<Gray>& a, const Image<Gray>& b, const Image<Gray>& c) {
    if (a.rows() != b.rows() || a.rows() != c.rows() ||
        a.cols() != b.cols() || a.cols() != c.cols())
        throw std::invalid_argument("MergeChannels: all channels must have the same size");
}

Image<BGR> MergeChannels::operator()(const Image<Gray>& b, const Image<Gray>& g,
                                      const Image<Gray>& r) const {
    check_sizes(b, g, r);
    cv::Mat result;
    cv::merge(std::vector<cv::Mat>{b.mat(), g.mat(), r.mat()}, result);
    return Image<BGR>(std::move(result));
}

Image<BGRA> MergeChannels::operator()(const Image<Gray>& b, const Image<Gray>& g,
                                       const Image<Gray>& r, const Image<Gray>& a) const {
    check_sizes(b, g, r);
    if (a.rows() != b.rows() || a.cols() != b.cols())
        throw std::invalid_argument("MergeChannels: all channels must have the same size");
    cv::Mat result;
    cv::merge(std::vector<cv::Mat>{b.mat(), g.mat(), r.mat(), a.mat()}, result);
    return Image<BGRA>(std::move(result));
}

} // namespace improc::core
