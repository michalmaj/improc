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

/**
 * @brief With probability `p`, erases a randomly placed rectangle filled with `value`.
 *
 * Area and aspect ratio of the erased region are sampled from the given ranges.
 * Up to 10 placement attempts are made; if all fail the image is returned unchanged.
 *
 * @code
 * auto erase = RandomErasing().p(0.5f).scale(0.02f, 0.33f).value(0);
 * img = img | erase.bind(rng);
 * @endcode
 */
struct RandomErasing : detail::BindMixin<RandomErasing> {
    /// @brief Sets the application probability (default: 0.5).
    /// @throws improc::ParameterError if `prob` is outside [0, 1].
    RandomErasing& p(float prob) {
        if (prob < 0.0f || prob > 1.0f)
            throw ParameterError{"p", "must be in [0, 1]", "RandomErasing"};
        p_ = prob; return *this;
    }
    /// @brief Sets the erased area as a fraction of the total image area (default: [0.02, 0.33]).
    /// @throws improc::ParameterError if `min_s` <= 0 or `max_s` > 1 or `min_s` > `max_s`.
    RandomErasing& scale(float min_s, float max_s) {
        if (min_s <= 0.0f || max_s > 1.0f || min_s > max_s)
            throw ParameterError{"scale", "must satisfy 0 < min <= max <= 1", "RandomErasing"};
        min_s_ = min_s; max_s_ = max_s; return *this;
    }
    /// @brief Sets the aspect ratio range of the erased rectangle (default: [0.3, 3.3]).
    /// @throws improc::ParameterError if `min_r` <= 0 or `min_r` > `max_r`.
    RandomErasing& ratio(float min_r, float max_r) {
        if (min_r <= 0.0f || min_r > max_r)
            throw ParameterError{"ratio", "must satisfy 0 < min <= max", "RandomErasing"};
        min_r_ = min_r; max_r_ = max_r; return *this;
    }
    /// @brief Sets the fill value for 8-bit images in [0, 255] (default: 0). Float images use `value/255`.
    /// @throws improc::ParameterError if `v` is outside [0, 255].
    RandomErasing& value(int v) {
        if (v < 0 || v > 255)
            throw ParameterError{"value", "must be in [0, 255]", "RandomErasing"};
        value_ = v; return *this;
    }

    template<AnyFormat Format>
    [[nodiscard]] Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
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

/**
 * @brief Randomly zeros out grid cells of size `unit_size × unit_size` with probability `ratio`.
 */
struct GridDropout : detail::BindMixin<GridDropout> {
    /// @brief Sets the per-cell dropout probability (default: 0.5). Must be in (0, 1).
    /// @throws improc::ParameterError if `r` is outside (0, 1).
    GridDropout& ratio(float r) {
        if (r <= 0.0f || r >= 1.0f)
            throw ParameterError{"ratio", "must be in (0, 1)", "GridDropout"};
        ratio_ = r; return *this;
    }
    /// @brief Sets the grid cell size in pixels (default: 32).
    /// @throws improc::ParameterError if `s` <= 0.
    GridDropout& unit_size(int s) {
        if (s <= 0)
            throw ParameterError{"unit_size", "must be positive", "GridDropout"};
        unit_size_ = s; return *this;
    }
    /// @brief Sets the fill value in [0, 255] (default: 0). Float images use `value / 255.0`.
    /// @throws improc::ParameterError if `v` is outside [0, 255].
    GridDropout& value(int v) {
        if (v < 0 || v > 255)
            throw ParameterError{"value", "must be in [0, 255]", "GridDropout"};
        value_ = v; return *this;
    }

    template<AnyFormat Format>
    [[nodiscard]] Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
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
