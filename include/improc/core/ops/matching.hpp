// include/improc/core/ops/matching.hpp
#pragma once
#include <vector>
#include <utility>
#include <opencv2/features2d.hpp>
#include "improc/core/ops/feature_detection.hpp"

namespace improc::core {

/**
 * @brief Result of a descriptor-matching operation.
 *
 * Plain struct with a public `matches` vector. Use `size()` and `empty()` for
 * convenience; iterate `matches` directly for per-match access.
 *
 * @code
 * MatchSet ms = MatchBF{desc1, desc2}();
 * for (auto& m : ms.matches)
 *     std::cout << m.distance << '\n';
 * @endcode
 */
struct MatchSet {
    std::vector<cv::DMatch> matches;

    std::size_t size()  const { return matches.size(); }
    bool        empty() const { return matches.empty(); }
};

/**
 * @brief Callable: brute-force descriptor matching.
 *
 * Norm type is auto-detected: NORM_HAMMING for CV_8U (ORB/AKAZE), NORM_L2 for CV_32F (SIFT).
 * Returns empty MatchSet for empty input descriptors.
 *
 * @throws ParameterError if max_distance < 0.
 *
 * @code
 * MatchSet ms = MatchBF{desc1, desc2}.cross_check(true)();
 * @endcode
 */
struct MatchBF {
    MatchBF(DescriptorSet lhs, DescriptorSet rhs)
        : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

    MatchBF& cross_check(bool v)   { cross_check_ = v; return *this; }

    MatchBF& max_distance(float v) {
        if (v < 0.0f)
            throw ParameterError{"max_distance", "must be >= 0", "MatchBF"};
        max_distance_ = v;
        return *this;
    }

    MatchSet operator()() const;

private:
    DescriptorSet lhs_;
    DescriptorSet rhs_;
    bool  cross_check_{false};
    float max_distance_{0.0f};
};

/**
 * @brief Callable: FLANN-based matching with Lowe ratio test.
 *
 * Runs knnMatch(k=2) and keeps matches where d1 < ratio_threshold * d2.
 * CV_32F descriptors only — throws ParameterError at call time for CV_8U.
 * Returns empty MatchSet for empty input descriptors.
 *
 * @throws ParameterError if ratio_threshold <= 0 || ratio_threshold > 1 (at setter).
 * @throws ParameterError if descriptors are not CV_32F (at call time).
 *
 * @code
 * MatchSet ms = MatchFlann{sift_desc1, sift_desc2}.ratio_threshold(0.75f)();
 * @endcode
 */
struct MatchFlann {
    MatchFlann(DescriptorSet lhs, DescriptorSet rhs)
        : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

    MatchFlann& ratio_threshold(float v) {
        if (v <= 0.0f || v > 1.0f)
            throw ParameterError{"ratio_threshold", "must be in (0, 1]", "MatchFlann"};
        ratio_threshold_ = v;
        return *this;
    }

    MatchSet operator()() const;

private:
    DescriptorSet lhs_;
    DescriptorSet rhs_;
    float ratio_threshold_{0.7f};
};

} // namespace improc::core
