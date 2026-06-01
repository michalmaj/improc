// include/improc/core/ops/arithmetic.hpp
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/core/ops/invert.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Computes the absolute difference between an image and a fixed matrix.
 */
struct AbsDiff {
    /// @brief Constructs with the subtrahend matrix.
    explicit AbsDiff(cv::Mat other) : other_(std::move(other)) {}

    /// @throws improc::ParameterError if sizes or types differ.
    template<AnyFormat F>
    Image<F> operator()(Image<F> img) const {
        if (img.mat().size() != other_.size() || img.mat().type() != other_.type())
            throw improc::ParameterError{"other", "must have the same size and type as lhs", "AbsDiff"};
        cv::Mat result;
        cv::absdiff(img.mat(), other_, result);
        return Image<F>(std::move(result));
    }

private:
    cv::Mat other_;
};

/**
 * @brief Applies element-wise bitwise AND with a fixed mask.
 * Requires an integer-format image (BGR, Gray, BGRA).
 */
struct BitwiseAnd {
    /// @brief Constructs with the mask matrix.
    explicit BitwiseAnd(cv::Mat other) : other_(std::move(other)) {}

    /// @throws improc::ParameterError if sizes or types differ.
    template<IntegerFormat F>
    Image<F> operator()(Image<F> img) const {
        if (img.mat().size() != other_.size() || img.mat().type() != other_.type())
            throw improc::ParameterError{"other", "must have the same size and type as lhs", "BitwiseAnd"};
        cv::Mat result;
        cv::bitwise_and(img.mat(), other_, result);
        return Image<F>(std::move(result));
    }

private:
    cv::Mat other_;
};

/**
 * @brief Applies element-wise bitwise OR with a fixed mask.
 * Requires an integer-format image (BGR, Gray, BGRA).
 */
struct BitwiseOr {
    /// @brief Constructs with the mask matrix.
    explicit BitwiseOr(cv::Mat other) : other_(std::move(other)) {}

    /// @throws improc::ParameterError if sizes or types differ.
    template<IntegerFormat F>
    Image<F> operator()(Image<F> img) const {
        if (img.mat().size() != other_.size() || img.mat().type() != other_.type())
            throw improc::ParameterError{"other", "must have the same size and type as lhs", "BitwiseOr"};
        cv::Mat result;
        cv::bitwise_or(img.mat(), other_, result);
        return Image<F>(std::move(result));
    }

private:
    cv::Mat other_;
};

/// @brief Alias for `Invert` — applies element-wise bitwise NOT.
using BitwiseNot = Invert;

/**
 * @brief Applies a 2-D convolution with a custom kernel via `cv::filter2D`.
 *
 * @code
 * cv::Mat sobel_x = (cv::Mat_<float>(3,3) << -1,0,1, -2,0,2, -1,0,1);
 * auto edges = Convolve(sobel_x)(img);
 * @endcode
 */
struct Convolve {
    /// @brief Constructs with the convolution kernel.
    /// @throws improc::ParameterError if kernel is empty.
    explicit Convolve(cv::Mat kernel) : kernel_(std::move(kernel)) {
        if (kernel_.empty())
            throw improc::ParameterError{"kernel", "must not be empty", "Convolve"};
    }

    /// @brief Sets the anchor point (default: {-1,-1} = kernel centre).
    Convolve& anchor(cv::Point a) { anchor_ = a; return *this; }
    /// @brief Sets the optional value added to the filtered result (default: 0).
    Convolve& delta(double d)     { delta_  = d; return *this; }
    /// @brief Sets the border extrapolation method (default: cv::BORDER_REFLECT_101).
    Convolve& border(int t)       { border_ = t; return *this; }

    template<AnyFormat F>
    Image<F> operator()(Image<F> img) const {
        cv::Mat result;
        cv::filter2D(img.mat(), result, -1, kernel_, anchor_, delta_, border_);
        return Image<F>(std::move(result));
    }

private:
    cv::Mat  kernel_;
    cv::Point anchor_ = {-1, -1};
    double   delta_   = 0.0;
    int      border_  = cv::BORDER_REFLECT_101;
};

/**
 * @brief Converts a signed/float matrix to displayable 8-bit via `cv::convertScaleAbs`.
 *
 * Primary use case: converting CV_16S Sobel/Laplacian output for visualisation.
 */
struct ConvertScaleAbs {
    /// @brief Sets the multiplicative scale factor (default: 1.0).
    ConvertScaleAbs& alpha(double a) { alpha_ = a; return *this; }
    /// @brief Sets the additive shift (default: 0.0).
    ConvertScaleAbs& beta(double b)  { beta_  = b; return *this; }

    /// @return Grayscale Image<Gray> (CV_8UC1).
    Image<Gray> operator()(const cv::Mat& src) const {
        cv::Mat result;
        cv::convertScaleAbs(src, result, alpha_, beta_);
        return Image<Gray>(std::move(result));
    }

private:
    double alpha_ = 1.0;
    double beta_  = 0.0;
};

/**
 * @brief Adds two images element-wise via `cv::add`.
 */
struct Add {
    /// @brief Constructs with the addend matrix.
    explicit Add(cv::Mat other) : other_(std::move(other)) {}

    /// @throws improc::ParameterError if sizes or types differ.
    template<AnyFormat F>
    Image<F> operator()(Image<F> img) const {
        if (img.mat().size() != other_.size() || img.mat().type() != other_.type())
            throw improc::ParameterError{"other", "must have the same size and type as lhs", "Add"};
        cv::Mat result;
        cv::add(img.mat(), other_, result);
        return Image<F>(std::move(result));
    }

private:
    cv::Mat other_;
};

/**
 * @brief Subtracts a fixed matrix from an image element-wise via `cv::subtract`.
 */
struct Subtract {
    /// @brief Constructs with the subtrahend matrix.
    explicit Subtract(cv::Mat other) : other_(std::move(other)) {}

    /// @throws improc::ParameterError if sizes or types differ.
    template<AnyFormat F>
    Image<F> operator()(Image<F> img) const {
        if (img.mat().size() != other_.size() || img.mat().type() != other_.type())
            throw improc::ParameterError{"other", "must have the same size and type as lhs", "Subtract"};
        cv::Mat result;
        cv::subtract(img.mat(), other_, result);
        return Image<F>(std::move(result));
    }

private:
    cv::Mat other_;
};

/**
 * @brief Multiplies two images element-wise via `cv::multiply`.
 */
struct Multiply {
    /// @brief Constructs with the multiplicand matrix.
    explicit Multiply(cv::Mat other) : other_(std::move(other)) {}
    /// @brief Sets an optional scale factor applied after multiplication (default: 1.0).
    Multiply& scale(double s) { scale_ = s; return *this; }

    /// @throws improc::ParameterError if sizes or types differ.
    template<AnyFormat F>
    Image<F> operator()(Image<F> img) const {
        if (img.mat().size() != other_.size() || img.mat().type() != other_.type())
            throw improc::ParameterError{"other", "must have the same size and type as lhs", "Multiply"};
        cv::Mat result;
        cv::multiply(img.mat(), other_, result, scale_);
        return Image<F>(std::move(result));
    }

private:
    cv::Mat other_;
    double  scale_ = 1.0;
};

/**
 * @brief Divides an image by a fixed matrix element-wise via `cv::divide`.
 */
struct Divide {
    /// @brief Constructs with the divisor matrix.
    explicit Divide(cv::Mat other) : other_(std::move(other)) {}
    /// @brief Sets an optional scale factor applied after division (default: 1.0).
    Divide& scale(double s) { scale_ = s; return *this; }

    /// @throws improc::ParameterError if sizes or types differ.
    template<AnyFormat F>
    Image<F> operator()(Image<F> img) const {
        if (img.mat().size() != other_.size() || img.mat().type() != other_.type())
            throw improc::ParameterError{"other", "must have the same size and type as lhs", "Divide"};
        cv::Mat result;
        cv::divide(img.mat(), other_, result, scale_);
        return Image<F>(std::move(result));
    }

private:
    cv::Mat other_;
    double  scale_ = 1.0;
};

} // namespace improc::core
