// include/improc/core/ops/analysis.hpp
#pragma once
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"

namespace improc::core {

struct IntegralResult {
    cv::Mat sum;     // CV_32SC1; size is (rows+1)×(cols+1)
    cv::Mat sq_sum;  // CV_64FC1; empty if with_sq_sum = false
};

struct IntegralImage {
    IntegralImage& with_sq_sum(bool b) { with_sq_sum_ = b; return *this; }

    IntegralResult operator()(const Image<Gray>& img) const;

private:
    bool with_sq_sum_ = false;
};

struct MinMaxLocResult {
    double    min_val, max_val;
    cv::Point min_loc, max_loc;
};

struct MinMaxLoc {
    MinMaxLocResult operator()(const Image<Gray>& img) const;
    MinMaxLocResult operator()(const cv::Mat& mat) const;
};

struct MeanStdDevResult {
    cv::Scalar mean;
    cv::Scalar stddev;
};

struct MeanStdDev {
    template<AnyFormat F>
    MeanStdDevResult operator()(const Image<F>& img) const {
        MeanStdDevResult r;
        cv::meanStdDev(img.mat(), r.mean, r.stddev);
        return r;
    }
};

struct CountNonZero {
    int operator()(const Image<Gray>& img) const;
};

enum class ReduceOp { Sum, Avg, Max, Min };

struct Reduce {
    Reduce& op(ReduceOp o) { op_  = o; return *this; }
    Reduce& dim(int d)     { dim_ = d; return *this; }

    cv::Mat operator()(const Image<Gray>& img) const;

private:
    ReduceOp op_  = ReduceOp::Sum;
    int      dim_ = 0;  // 0 = reduce rows → single row; 1 = reduce cols → single col
};

} // namespace improc::core
