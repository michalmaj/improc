// src/core/ops/matching.cpp
#include "improc/core/ops/matching.hpp"
#include <algorithm>

namespace improc::core {

MatchSet MatchBF::operator()() const {
    if (lhs_.descriptors.empty() || rhs_.descriptors.empty())
        return MatchSet{};

    int norm_type = (lhs_.descriptors.type() == CV_8U)
                    ? cv::NORM_HAMMING : cv::NORM_L2;
    auto matcher = cv::BFMatcher::create(norm_type, cross_check_);

    MatchSet result;
    matcher->match(lhs_.descriptors, rhs_.descriptors, result.matches);

    if (max_distance_ > 0.0f)
        std::erase_if(result.matches,
            [d = max_distance_](const cv::DMatch& m) { return m.distance > d; });

    return result;
}

MatchSet MatchFlann::operator()() const {
    if (lhs_.descriptors.empty() || rhs_.descriptors.empty())
        return MatchSet{};

    if (lhs_.descriptors.type() != CV_32F || rhs_.descriptors.type() != CV_32F)
        throw ParameterError{"descriptors",
            "FLANN requires float descriptors (SIFT)", "MatchFlann"};

    auto matcher = cv::FlannBasedMatcher::create();
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(lhs_.descriptors, rhs_.descriptors, knn_matches, 2);

    MatchSet result;
    for (auto& pair : knn_matches) {
        if (pair.size() < 2) continue;
        if (pair[0].distance < ratio_threshold_ * pair[1].distance)
            result.matches.push_back(pair[0]);
    }
    return result;
}

} // namespace improc::core
