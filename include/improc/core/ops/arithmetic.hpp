// include/improc/core/ops/arithmetic.hpp
#pragma once
#include <stdexcept>
#include <opencv2/core.hpp>
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

} // namespace improc::core
