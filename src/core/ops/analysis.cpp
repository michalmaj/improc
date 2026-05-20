// src/core/ops/analysis.cpp
#include "improc/core/ops/analysis.hpp"
#include <opencv2/imgproc.hpp>

namespace improc::core {

IntegralResult IntegralImage::operator()(const Image<Gray>& img) const {
    IntegralResult r;
    if (with_sq_sum_)
        cv::integral(img.mat(), r.sum, r.sq_sum);
    else
        cv::integral(img.mat(), r.sum);
    return r;
}

static MinMaxLocResult run_minmaxloc(const cv::Mat& m) {
    MinMaxLocResult r;
    cv::minMaxLoc(m, &r.min_val, &r.max_val, &r.min_loc, &r.max_loc);
    return r;
}

MinMaxLocResult MinMaxLoc::operator()(const Image<Gray>& img) const {
    return run_minmaxloc(img.mat());
}

MinMaxLocResult MinMaxLoc::operator()(const cv::Mat& mat) const {
    return run_minmaxloc(mat);
}

int CountNonZero::operator()(const Image<Gray>& img) const {
    return cv::countNonZero(img.mat());
}

cv::Mat Reduce::operator()(const Image<Gray>& img) const {
    int cv_op;
    switch (op_) {
        case ReduceOp::Sum: cv_op = cv::REDUCE_SUM; break;
        case ReduceOp::Avg: cv_op = cv::REDUCE_AVG; break;
        case ReduceOp::Max: cv_op = cv::REDUCE_MAX; break;
        case ReduceOp::Min: cv_op = cv::REDUCE_MIN; break;
    }
    cv::Mat result;
    cv::reduce(img.mat(), result, dim_, cv_op, CV_32SC1);
    return result;
}

} // namespace improc::core
