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

Image<Flow> DenseFarnebackFlow::operator()(const Image<Gray>& prev,
                                            const Image<Gray>& next) const {
    if (prev.rows() != next.rows() || prev.cols() != next.cols())
        throw std::invalid_argument("DenseFarnebackFlow: prev and next must have the same size");
    cv::Mat flow;
    cv::calcOpticalFlowFarneback(prev.mat(), next.mat(), flow,
                                  pyr_scale_, levels_, win_size_,
                                  iterations_, poly_n_, poly_sigma_, 0);
    return Image<Flow>(std::move(flow));
}

} // namespace improc::core
