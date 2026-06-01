// src/core/ops/optical_flow.cpp
#include "improc/core/ops/optical_flow.hpp"
#include "improc/exceptions.hpp"
#include <stdexcept>

namespace improc::core {

SparseLKFlowResult SparseLKFlow::operator()(const Image<Gray>& prev,
                                             const Image<Gray>& next,
                                             const std::vector<cv::Point2f>& prev_pts) const {
    if (prev_pts.empty())
        throw improc::ParameterError{"prev_pts", "must not be empty", "SparseLKFlow"};
    if (prev.rows() != next.rows() || prev.cols() != next.cols())
        throw improc::ParameterError{"prev", "prev and next must have the same size", "SparseLKFlow"};

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
        throw improc::ParameterError{"prev", "prev and next must have the same size", "DenseFarnebackFlow"};
    cv::Mat flow;
    cv::calcOpticalFlowFarneback(prev.mat(), next.mat(), flow,
                                  pyr_scale_, levels_, win_size_,
                                  iterations_, poly_n_, poly_sigma_, 0);
    return Image<Flow>(std::move(flow));
}

Image<Flow> DenseDISFlow::operator()(const Image<Gray>& prev,
                                      const Image<Gray>& next) const {
    if (prev.rows() != next.rows() || prev.cols() != next.cols())
        throw improc::ParameterError{"prev", "prev and next must have the same size", "DenseDISFlow"};
    int preset_flag;
    switch (preset_) {
        case Preset::UltraFast: preset_flag = cv::DISOpticalFlow::PRESET_ULTRAFAST; break;
        case Preset::Fast:      preset_flag = cv::DISOpticalFlow::PRESET_FAST;      break;
        case Preset::Medium:    preset_flag = cv::DISOpticalFlow::PRESET_MEDIUM;    break;
        default:                std::unreachable();
    }
    auto dis = cv::DISOpticalFlow::create(preset_flag);
    cv::Mat flow;
    dis->calc(prev.mat(), next.mat(), flow);
    return Image<Flow>(std::move(flow));
}

} // namespace improc::core
