// include/improc/ml/augment/blur.hpp
#pragma once

#include <algorithm>
#include <random>
#include <vector>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/ml/augment/detail.hpp"
#include "improc/exceptions.hpp"
#include "improc/ml/segmented.hpp"

namespace improc::ml {

using improc::core::AnyFormat;
using improc::core::Image;

/**
 * @brief Randomly applies one of {Gaussian, Median, Bilateral} blur with a randomly sampled odd kernel size.
 *
 * @code
 * std::mt19937 rng(42);
 * auto aug = RandomBlur()
 *     .types({RandomBlur::Type::Gaussian, RandomBlur::Type::Median})
 *     .kernel_size(3, 7);
 * img = img | aug.bind(rng);
 * @endcode
 */
struct RandomBlur : detail::BindMixin<RandomBlur> {
    /// @brief Blur algorithm choices.
    enum class Type {
        Gaussian,  ///< Gaussian blur via cv::GaussianBlur.
        Median,    ///< Median blur via cv::medianBlur.
        Bilateral  ///< Bilateral filter via cv::bilateralFilter.
    };

    /// @brief Sets the blur type pool to sample from (default: {Gaussian, Median}).
    /// @throws improc::ParameterError if `t` is empty.
    RandomBlur& types(std::vector<Type> t) {
        if (t.empty())
            throw ParameterError{"types", "must not be empty", "RandomBlur"};
        types_ = std::move(t); return *this;
    }
    /// @brief Sets the odd kernel-size range in [3, 31] (default: [3, 7]).
    /// @throws improc::ParameterError if values are out of range or even.
    RandomBlur& kernel_size(int min_k, int max_k) {
        if (min_k < 3 || max_k > 31 || min_k % 2 == 0 || max_k % 2 == 0 || min_k > max_k)
            throw ParameterError{"kernel_size", "must be odd, in [3, 31], min <= max", "RandomBlur"};
        min_k_ = min_k; max_k_ = max_k; return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        constexpr bool is_float = improc::core::FormatTraits<Format>::is_float;
        if constexpr (is_float) {
            bool has_bilateral = std::find(types_.begin(), types_.end(), Type::Bilateral) != types_.end();
            if (has_bilateral)
                throw ParameterError{"format", "Bilateral requires 8-bit format", "RandomBlur"};
        }
        // Sample type
        std::uniform_int_distribution<int> type_d(0, static_cast<int>(types_.size()) - 1);
        Type chosen = types_[type_d(rng)];

        std::uniform_int_distribution<int> ks_d(0, (max_k_ - min_k_) / 2);
        int ks = min_k_ + ks_d(rng) * 2;

        cv::Mat dst;
        switch (chosen) {
            case Type::Gaussian:
                cv::GaussianBlur(img.mat(), dst, cv::Size(ks, ks), 0);
                break;
            case Type::Median:
                cv::medianBlur(img.mat(), dst, ks);
                break;
            case Type::Bilateral:
                cv::bilateralFilter(img.mat(), dst, ks, 75, 75);
                break;
            default: std::unreachable();
        }
        return Image<Format>(std::move(dst));
    }

    template<AnyFormat Format>
    SegmentedImage<Format> operator()(SegmentedImage<Format> seg, std::mt19937& rng) const {
        seg.image = (*this)(std::move(seg.image), rng);
        return seg;
    }

private:
    std::vector<Type> types_  = {Type::Gaussian, Type::Median};
    int               min_k_  = 3;
    int               max_k_  = 7;
};

/**
 * @brief Randomly applies unsharp masking to enhance image sharpness.
 * Applies with probability `p`; strength is sampled uniformly from [min_s, max_s].
 */
struct RandomSharpness : detail::BindMixin<RandomSharpness> {
    /// @brief Sets the sharpness strength range (default: [0.0, 1.0]).
    /// @throws improc::ParameterError if `min_s` < 0 or `min_s` > `max_s`.
    RandomSharpness& range(float min_s, float max_s) {
        if (min_s < 0.0f || min_s > max_s)
            throw ParameterError{"range", "must satisfy 0 <= min <= max", "RandomSharpness"};
        min_s_ = min_s; max_s_ = max_s; return *this;
    }
    /// @brief Sets the application probability (default: 0.5).
    /// @throws improc::ParameterError if `prob` is outside [0, 1].
    RandomSharpness& p(float prob) {
        if (prob < 0.0f || prob > 1.0f)
            throw ParameterError{"p", "must be in [0, 1]", "RandomSharpness"};
        p_ = prob; return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        std::bernoulli_distribution d(p_);
        if (!d(rng)) return img;
        std::uniform_real_distribution<float> s_d(min_s_, max_s_);
        float strength = s_d(rng);
        cv::Mat blurred;
        cv::GaussianBlur(img.mat(), blurred, cv::Size(3, 3), 0);
        cv::Mat src_f, blur_f;
        img.mat().convertTo(src_f, CV_32F);
        blurred.convertTo(blur_f, CV_32F);
        cv::Mat dst_f = src_f + strength * (src_f - blur_f);
        if constexpr (improc::core::FormatTraits<Format>::is_float) {
            cv::min(dst_f, 1.0, dst_f);
            cv::max(dst_f, 0.0, dst_f);
        } else {
            cv::min(dst_f, 255.0, dst_f);
            cv::max(dst_f, 0.0,   dst_f);
        }
        cv::Mat dst;
        dst_f.convertTo(dst, img.mat().type());
        return Image<Format>(std::move(dst));
    }

    template<AnyFormat Format>
    SegmentedImage<Format> operator()(SegmentedImage<Format> seg, std::mt19937& rng) const {
        seg.image = (*this)(std::move(seg.image), rng);
        return seg;
    }

private:
    float min_s_ = 0.0f;
    float max_s_ = 1.0f;
    float p_     = 0.5f;
};

} // namespace improc::ml
