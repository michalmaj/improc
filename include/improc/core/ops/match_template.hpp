// include/improc/core/ops/match_template.hpp
#pragma once

#include <utility>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

/**
 * @brief Slides a template over an image and returns the best match location.
 *
 * @code
 * auto [pt, score] = MatchTemplate().method(cv::TM_CCOEFF_NORMED)(img, templ);
 * @endcode
 */
struct MatchTemplate {
    /// @brief Sets the matching method (default: cv::TM_CCOEFF_NORMED).
    MatchTemplate& method(int m);

    /// @return {best_match_top_left, match_score}.
    [[nodiscard]] std::pair<cv::Point, double> operator()(const Image<BGR>& img,
                                             const Image<BGR>& templ) const;

private:
    int method_ = cv::TM_CCOEFF_NORMED;
};

} // namespace improc::core
