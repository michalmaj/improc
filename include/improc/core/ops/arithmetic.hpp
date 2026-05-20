// include/improc/core/ops/arithmetic.hpp
#pragma once
#include <stdexcept>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/core/ops/invert.hpp"

namespace improc::core {

struct AbsDiff {
    explicit AbsDiff(cv::Mat other) : other_(std::move(other)) {}

    template<AnyFormat F>
    Image<F> operator()(Image<F> img) const {
        if (img.mat().size() != other_.size() || img.mat().type() != other_.type())
            throw std::invalid_argument("AbsDiff: images must have the same size and type");
        cv::Mat result;
        cv::absdiff(img.mat(), other_, result);
        return Image<F>(std::move(result));
    }

private:
    cv::Mat other_;
};

struct BitwiseAnd {
    explicit BitwiseAnd(cv::Mat other) : other_(std::move(other)) {}

    template<IntegerFormat F>
    Image<F> operator()(Image<F> img) const {
        if (img.mat().size() != other_.size() || img.mat().type() != other_.type())
            throw std::invalid_argument("BitwiseAnd: images must have the same size and type");
        cv::Mat result;
        cv::bitwise_and(img.mat(), other_, result);
        return Image<F>(std::move(result));
    }

private:
    cv::Mat other_;
};

struct BitwiseOr {
    explicit BitwiseOr(cv::Mat other) : other_(std::move(other)) {}

    template<IntegerFormat F>
    Image<F> operator()(Image<F> img) const {
        if (img.mat().size() != other_.size() || img.mat().type() != other_.type())
            throw std::invalid_argument("BitwiseOr: images must have the same size and type");
        cv::Mat result;
        cv::bitwise_or(img.mat(), other_, result);
        return Image<F>(std::move(result));
    }

private:
    cv::Mat other_;
};

using BitwiseNot = Invert;

struct Convolve {
    explicit Convolve(cv::Mat kernel) : kernel_(std::move(kernel)) {
        if (kernel_.empty())
            throw std::invalid_argument("Convolve: kernel must not be empty");
    }

    Convolve& anchor(cv::Point a) { anchor_ = a; return *this; }
    Convolve& delta(double d)     { delta_  = d; return *this; }
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

struct ConvertScaleAbs {
    ConvertScaleAbs& alpha(double a) { alpha_ = a; return *this; }
    ConvertScaleAbs& beta(double b)  { beta_  = b; return *this; }

    Image<Gray> operator()(const cv::Mat& src) const {
        cv::Mat result;
        cv::convertScaleAbs(src, result, alpha_, beta_);
        return Image<Gray>(std::move(result));
    }

private:
    double alpha_ = 1.0;
    double beta_  = 0.0;
};

struct Add {
    explicit Add(cv::Mat other) : other_(std::move(other)) {}

    template<AnyFormat F>
    Image<F> operator()(Image<F> img) const {
        if (img.mat().size() != other_.size() || img.mat().type() != other_.type())
            throw std::invalid_argument("Add: images must have the same size and type");
        cv::Mat result;
        cv::add(img.mat(), other_, result);
        return Image<F>(std::move(result));
    }

private:
    cv::Mat other_;
};

struct Subtract {
    explicit Subtract(cv::Mat other) : other_(std::move(other)) {}

    template<AnyFormat F>
    Image<F> operator()(Image<F> img) const {
        if (img.mat().size() != other_.size() || img.mat().type() != other_.type())
            throw std::invalid_argument("Subtract: images must have the same size and type");
        cv::Mat result;
        cv::subtract(img.mat(), other_, result);
        return Image<F>(std::move(result));
    }

private:
    cv::Mat other_;
};

struct Multiply {
    explicit Multiply(cv::Mat other) : other_(std::move(other)) {}
    Multiply& scale(double s) { scale_ = s; return *this; }

    template<AnyFormat F>
    Image<F> operator()(Image<F> img) const {
        if (img.mat().size() != other_.size() || img.mat().type() != other_.type())
            throw std::invalid_argument("Multiply: images must have the same size and type");
        cv::Mat result;
        cv::multiply(img.mat(), other_, result, scale_);
        return Image<F>(std::move(result));
    }

private:
    cv::Mat other_;
    double  scale_ = 1.0;
};

struct Divide {
    explicit Divide(cv::Mat other) : other_(std::move(other)) {}
    Divide& scale(double s) { scale_ = s; return *this; }

    template<AnyFormat F>
    Image<F> operator()(Image<F> img) const {
        if (img.mat().size() != other_.size() || img.mat().type() != other_.type())
            throw std::invalid_argument("Divide: images must have the same size and type");
        cv::Mat result;
        cv::divide(img.mat(), other_, result, scale_);
        return Image<F>(std::move(result));
    }

private:
    cv::Mat other_;
    double  scale_ = 1.0;
};

} // namespace improc::core
