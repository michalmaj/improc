// include/improc/ml/augment/noise.hpp
#pragma once

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/ml/augment/detail.hpp"

namespace improc::ml {

using improc::core::AnyFormat;
using improc::core::Image;

struct RandomGaussianNoise : detail::BindMixin<RandomGaussianNoise> {
    RandomGaussianNoise& std_dev(float low, float high) {
        if (low < 0.0f)
            throw std::invalid_argument("RandomGaussianNoise: std_dev low must be >= 0");
        if (low > high)
            throw std::invalid_argument("RandomGaussianNoise: std_dev low must be <= high");
        std_low_ = low; std_high_ = high; return *this;
    }
    RandomGaussianNoise& mean(float m) { mean_ = m; return *this; }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        std::uniform_real_distribution<float> d(std_low_, std_high_);
        float std_dev = d(rng);
        const int ch = img.mat().channels();
        cv::Mat noise(img.mat().size(), CV_32FC(ch));
        cv::randn(noise, cv::Scalar::all(mean_), cv::Scalar::all(std_dev));
        cv::Mat src_f, dst;
        img.mat().convertTo(src_f, CV_32FC(ch));
        src_f += noise;
        cv::min(src_f, 255.0, src_f);
        cv::max(src_f, 0.0,   src_f);
        src_f.convertTo(dst, img.mat().type());
        try {
            return Image<Format>(std::move(dst));
        } catch (const cv::Exception& e) {
            throw std::runtime_error("RandomGaussianNoise: " + std::string(e.what()));
        }
    }

private:
    float std_low_  =  5.0f;
    float std_high_ = 20.0f;
    float mean_     =  0.0f;
};

struct RandomSaltAndPepper : detail::BindMixin<RandomSaltAndPepper> {
    RandomSaltAndPepper& p(float prob) {
        if (prob < 0.0f || prob > 1.0f)
            throw std::invalid_argument("RandomSaltAndPepper: p must be in [0, 1]");
        p_ = prob; return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        cv::Mat dst = img.mat().clone();
        std::bernoulli_distribution noise_dist(p_);
        std::bernoulli_distribution salt_dist(0.5);
        const int channels = dst.channels();
        const int depth    = dst.depth();

        for (int r = 0; r < dst.rows; ++r) {
            for (int c = 0; c < dst.cols; ++c) {
                if (!noise_dist(rng)) continue;
                bool is_salt = salt_dist(rng);
                if (depth == CV_8U) {
                    uchar val = is_salt ? 255 : 0;
                    auto* ptr = dst.ptr<uchar>(r) + c * channels;
                    for (int ch = 0; ch < channels; ++ch) ptr[ch] = val;
                } else if (depth == CV_32F) {
                    float val = is_salt ? 1.0f : 0.0f;
                    auto* ptr = dst.ptr<float>(r) + c * channels;
                    for (int ch = 0; ch < channels; ++ch) ptr[ch] = val;
                }
            }
        }
        return Image<Format>(std::move(dst));
    }

private:
    float p_ = 0.05f;
};

} // namespace improc::ml
