// include/improc/core/ops/analysis.hpp
#pragma once
#include <stdexcept>
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"

namespace improc::core {

/**
 * @brief Result of `IntegralImage`.
 */
struct IntegralResult {
    cv::Mat sum;     ///< CV_32SC1 integral image; size is (rows+1)×(cols+1).
    cv::Mat sq_sum;  ///< CV_64FC1 squared integral image; empty if with_sq_sum was false.
};

/**
 * @brief Computes the integral (summed-area) image of a grayscale image.
 */
struct IntegralImage {
    /// @brief Also computes the squared integral image (default: false).
    IntegralImage& with_sq_sum(bool b) { with_sq_sum_ = b; return *this; }

    /// @brief Computes and returns the integral image.
    IntegralResult operator()(const Image<Gray>& img) const;

private:
    bool with_sq_sum_ = false;
};

/**
 * @brief Result of `MinMaxLoc`: minimum and maximum pixel values and their locations.
 */
struct MinMaxLocResult {
    double    min_val, max_val; ///< Minimum and maximum pixel values.
    cv::Point min_loc, max_loc; ///< Pixel coordinates of the minimum and maximum.
};

/**
 * @brief Finds the minimum and maximum values and their locations in a grayscale image or matrix.
 */
struct MinMaxLoc {
    /// @brief Finds min/max in a `Gray` image.
    MinMaxLocResult operator()(const Image<Gray>& img) const;
    /// @brief Finds min/max in an arbitrary `cv::Mat`.
    MinMaxLocResult operator()(const cv::Mat& mat) const;
};

/**
 * @brief Result of `MeanStdDev`: per-channel mean and standard deviation.
 */
struct MeanStdDevResult {
    cv::Scalar mean;   ///< Per-channel mean (up to 4 channels).
    cv::Scalar stddev; ///< Per-channel standard deviation.
};

/**
 * @brief Computes per-channel mean and standard deviation for any image format.
 */
struct MeanStdDev {
    template<AnyFormat F>
    MeanStdDevResult operator()(const Image<F>& img) const {
        MeanStdDevResult r;
        cv::meanStdDev(img.mat(), r.mean, r.stddev);
        return r;
    }
};

/**
 * @brief Counts non-zero pixels in a grayscale image.
 */
struct CountNonZero {
    /// @return Number of non-zero pixels.
    int operator()(const Image<Gray>& img) const;
};

/// @brief Reduction operation used by `Reduce`.
enum class ReduceOp { Sum, Avg, Max, Min };

/**
 * @brief Reduces a grayscale image along rows or columns using a `ReduceOp`.
 */
struct Reduce {
    /// @brief Sets the reduction operation (default: Sum).
    Reduce& op(ReduceOp o) { op_  = o; return *this; }
    /// @brief Sets the reduction dimension: 0 = reduce rows → single row; 1 = reduce cols → single col.
    /// @throws std::invalid_argument if dim is not 0 or 1.
    Reduce& dim(int d) {
        if (d != 0 && d != 1)
            throw std::invalid_argument("Reduce: dim must be 0 (reduce rows) or 1 (reduce cols)");
        dim_ = d;
        return *this;
    }

    /// @return Single-row or single-column cv::Mat.
    cv::Mat operator()(const Image<Gray>& img) const;

private:
    ReduceOp op_  = ReduceOp::Sum;
    int      dim_ = 0;  // 0 = reduce rows → single row; 1 = reduce cols → single col
};

} // namespace improc::core
