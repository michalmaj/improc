#include "improc/core/ops/optical_flow.hpp"
#include <stdexcept>

namespace improc::core {

SparseLKFlowResult SparseLKFlow::operator()(const Image<Gray>& prev,
                                             const Image<Gray>& next,
                                             const std::vector<cv::Point2f>& prev_pts) const {
    if (prev_pts.empty())
        throw std::invalid_argument("SparseLKFlow: prev_pts must not be empty");
    if (prev.rows() != next.rows() || prev.cols() != next.cols())
        throw std::invalid_argument("SparseLKFlow: prev and next must have the same size");

    SparseLKFlowResult result;
    cv::calcOpticalFlowPyrLK(
        prev.mat(), next.mat(), prev_pts,
        result.points, result.status, result.error,
        win_size_, max_level_,
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                         max_iter_, epsilon_));
    return result;
}

} // namespace improc::core
