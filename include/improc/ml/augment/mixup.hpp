// include/improc/ml/augment/mixup.hpp
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/ml/labeled.hpp"
#include "improc/exceptions.hpp"

namespace improc::ml {

using improc::core::AnyFormat;
using improc::core::Image;

namespace detail {
inline float sample_beta(float alpha, std::mt19937& rng) {
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    float x = gamma(rng);
    float y = gamma(rng);
    float s = x + y;
    return (s > 0.0f) ? x / s : 0.5f;
}
} // namespace detail

struct MixUp {
    MixUp& alpha(float a) {
        if (a <= 0.0f)
            throw ParameterError{"alpha", "must be > 0", "MixUp"};
        alpha_ = a; return *this;
    }
    MixUp& p(float prob) {
        if (prob < 0.0f || prob > 1.0f)
            throw ParameterError{"p", "must be in [0, 1]", "MixUp"};
        p_ = prob; return *this;
    }

    template<AnyFormat F>
    LabeledImage<F> operator()(LabeledImage<F> a,
                                const LabeledImage<F>& b,
                                std::mt19937& rng) const {
        if (a.image.mat().size() != b.image.mat().size())
            throw ParameterError{"image", "primary and secondary must have the same size", "MixUp"};
        if (a.label.empty() || b.label.empty())
            throw ParameterError{"label", "label vectors must not be empty", "MixUp"};
        if (a.label.size() != b.label.size())
            throw ParameterError{"label", "label vectors must have the same size", "MixUp"};

        std::bernoulli_distribution apply(p_);
        if (!apply(rng)) return a;

        float lambda = detail::sample_beta(alpha_, rng);

        cv::Mat result;
        cv::addWeighted(a.image.mat(), static_cast<double>(lambda),
                        b.image.mat(), static_cast<double>(1.0f - lambda),
                        0.0, result);

        std::vector<float> label(a.label.size());
        for (std::size_t i = 0; i < label.size(); ++i)
            label[i] = lambda * a.label[i] + (1.0f - lambda) * b.label[i];

        return LabeledImage<F>{Image<F>(result), std::move(label)};
    }

private:
    float alpha_ = 0.4f;
    float p_     = 1.0f;
};

struct CutMix {
    CutMix& alpha(float a) {
        if (a <= 0.0f)
            throw ParameterError{"alpha", "must be > 0", "CutMix"};
        alpha_ = a; return *this;
    }
    CutMix& p(float prob) {
        if (prob < 0.0f || prob > 1.0f)
            throw ParameterError{"p", "must be in [0, 1]", "CutMix"};
        p_ = prob; return *this;
    }

    template<AnyFormat F>
    LabeledImage<F> operator()(LabeledImage<F> a,
                                const LabeledImage<F>& b,
                                std::mt19937& rng) const {
        if (a.image.mat().size() != b.image.mat().size())
            throw ParameterError{"image", "primary and secondary must have the same size", "CutMix"};
        if (a.label.empty() || b.label.empty())
            throw ParameterError{"label", "label vectors must not be empty", "CutMix"};
        if (a.label.size() != b.label.size())
            throw ParameterError{"label", "label vectors must have the same size", "CutMix"};

        std::bernoulli_distribution apply(p_);
        if (!apply(rng)) return a;

        float lambda = detail::sample_beta(alpha_, rng);
        const int W = a.image.cols();
        const int H = a.image.rows();

        int cut_w = static_cast<int>(static_cast<float>(W) * std::sqrt(1.0f - lambda));
        int cut_h = static_cast<int>(static_cast<float>(H) * std::sqrt(1.0f - lambda));
        cut_w = std::clamp(cut_w, 0, W);
        cut_h = std::clamp(cut_h, 0, H);

        cv::Mat result = a.image.mat().clone();

        if (cut_w == 0 || cut_h == 0)
            return LabeledImage<F>{Image<F>(result), a.label};

        std::uniform_int_distribution<int> cx_dist(0, W - 1);
        std::uniform_int_distribution<int> cy_dist(0, H - 1);
        int cx = cx_dist(rng);
        int cy = cy_dist(rng);

        int x1 = std::clamp(cx - cut_w / 2, 0, W - cut_w);
        int y1 = std::clamp(cy - cut_h / 2, 0, H - cut_h);
        cv::Rect patch(x1, y1, cut_w, cut_h);
        b.image.mat()(patch).copyTo(result(patch));

        float lambda_actual = 1.0f - (static_cast<float>(cut_w) * static_cast<float>(cut_h)) /
                                      (static_cast<float>(W) * static_cast<float>(H));
        std::vector<float> label(a.label.size());
        for (std::size_t i = 0; i < label.size(); ++i)
            label[i] = lambda_actual * a.label[i] + (1.0f - lambda_actual) * b.label[i];

        return LabeledImage<F>{Image<F>(result), std::move(label)};
    }

private:
    float alpha_ = 1.0f;
    float p_     = 1.0f;
};

} // namespace improc::ml
