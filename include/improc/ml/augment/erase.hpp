// include/improc/ml/augment/erase.hpp
#pragma once

#include <cmath>
#include <random>
#include <type_traits>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/ml/augment/detail.hpp"
#include "improc/exceptions.hpp"

namespace improc::ml {

using improc::core::AnyFormat;
using improc::core::Image;

struct RandomErasing : detail::BindMixin<RandomErasing> {
    RandomErasing& p(float prob) {
        if (prob < 0.0f || prob > 1.0f)
            throw ParameterError{"p", "must be in [0, 1]", "RandomErasing"};
        p_ = prob; return *this;
    }
    RandomErasing& scale(float min_s, float max_s) {
        if (min_s <= 0.0f || max_s > 1.0f || min_s > max_s)
            throw ParameterError{"scale", "must satisfy 0 < min <= max <= 1", "RandomErasing"};
        min_s_ = min_s; max_s_ = max_s; return *this;
    }
    RandomErasing& ratio(float min_r, float max_r) {
        if (min_r <= 0.0f || min_r > max_r)
            throw ParameterError{"ratio", "must satisfy 0 < min <= max", "RandomErasing"};
        min_r_ = min_r; max_r_ = max_r; return *this;
    }
    RandomErasing& value(int v) {
        if (v < 0 || v > 255)
            throw ParameterError{"value", "must be in [0, 255]", "RandomErasing"};
        value_ = v; return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        std::bernoulli_distribution d(p_);
        if (!d(rng)) return img;
        int W = img.cols(), H = img.rows();
        float area = static_cast<float>(W * H);
        std::uniform_real_distribution<float> area_d(min_s_ * area, max_s_ * area);
        std::uniform_real_distribution<float> ratio_d(min_r_, max_r_);
        cv::Mat dst = img.mat().clone();
        for (int attempt = 0; attempt < 10; ++attempt) {
            float target_area = area_d(rng);
            float ratio = ratio_d(rng);
            int w = static_cast<int>(std::round(std::sqrt(target_area * ratio)));
            int h = static_cast<int>(std::round(std::sqrt(target_area / ratio)));
            if (w <= 0 || h <= 0 || w > W || h > H) continue;
            std::uniform_int_distribution<int> xd(0, W - w);
            std::uniform_int_distribution<int> yd(0, H - h);
            int x = xd(rng), y = yd(rng);
            if constexpr (improc::core::FormatTraits<Format>::is_float)
                dst(cv::Rect(x, y, w, h)).setTo(cv::Scalar::all(value_ / 255.0));
            else
                dst(cv::Rect(x, y, w, h)).setTo(cv::Scalar::all(value_));
            return Image<Format>(std::move(dst));
        }
        return Image<Format>(std::move(dst));
    }

private:
    float p_     = 0.5f;
    float min_s_ = 0.02f;
    float max_s_ = 0.33f;
    float min_r_ = 0.3f;
    float max_r_ = 3.3f;
    int   value_ = 0;
};

struct GridDropout : detail::BindMixin<GridDropout> {
    GridDropout& ratio(float r) {
        if (r <= 0.0f || r >= 1.0f)
            throw ParameterError{"ratio", "must be in (0, 1)", "GridDropout"};
        ratio_ = r; return *this;
    }
    GridDropout& unit_size(int s) {
        if (s <= 0)
            throw ParameterError{"unit_size", "must be positive", "GridDropout"};
        unit_size_ = s; return *this;
    }
    GridDropout& value(int v) {
        if (v < 0 || v > 255)
            throw ParameterError{"value", "must be in [0, 255]", "GridDropout"};
        value_ = v; return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        cv::Mat dst = img.mat().clone();
        std::bernoulli_distribution d(ratio_);
        int W = img.cols(), H = img.rows();
        for (int y = 0; y < H; y += unit_size_) {
            for (int x = 0; x < W; x += unit_size_) {
                if (d(rng)) {
                    int w = std::min(unit_size_, W - x);
                    int h = std::min(unit_size_, H - y);
                    if constexpr (improc::core::FormatTraits<Format>::is_float)
                        dst(cv::Rect(x, y, w, h)).setTo(cv::Scalar::all(value_ / 255.0));
                    else
                        dst(cv::Rect(x, y, w, h)).setTo(cv::Scalar::all(value_));
                }
            }
        }
        return Image<Format>(std::move(dst));
    }

private:
    float ratio_     = 0.5f;
    int   unit_size_ = 32;
    int   value_     = 0;
};

} // namespace improc::ml
