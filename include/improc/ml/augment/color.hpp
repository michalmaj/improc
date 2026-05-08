// include/improc/ml/augment/color.hpp
#pragma once

#include <random>
#include <cmath>
#include <type_traits>
#include <vector>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/ml/augment/detail.hpp"
#include "improc/exceptions.hpp"

namespace improc::ml {

using improc::core::AnyFormat;
using improc::core::BGRFormat;
using improc::core::Image;

struct RandomBrightness : detail::BindMixin<RandomBrightness> {
    RandomBrightness& range(float low, float high) {
        if (low <= 0.0f)
            throw ParameterError{"low", "must be > 0", "RandomBrightness"};
        if (low > high)
            throw ParameterError{"low", "must be <= high", "RandomBrightness"};
        low_ = low; high_ = high; return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        std::uniform_real_distribution<float> d(low_, high_);
        float factor = d(rng);
        cv::Mat src_f, dst;
        img.mat().convertTo(src_f, CV_32F);
        src_f *= factor;
        cv::min(src_f, 255.0, src_f);
        cv::max(src_f, 0.0,   src_f);
        src_f.convertTo(dst, img.mat().type());
        return Image<Format>(std::move(dst));
    }

private:
    float low_  = 0.8f;
    float high_ = 1.2f;
};

struct RandomContrast : detail::BindMixin<RandomContrast> {
    RandomContrast& range(float low, float high) {
        if (low <= 0.0f)
            throw ParameterError{"low", "must be > 0", "RandomContrast"};
        if (low > high)
            throw ParameterError{"low", "must be <= high", "RandomContrast"};
        low_ = low; high_ = high; return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        std::uniform_real_distribution<float> d(low_, high_);
        float alpha = d(rng);
        cv::Scalar mean = cv::mean(img.mat());
        cv::Mat src_f, dst;
        img.mat().convertTo(src_f, CV_32F);
        src_f = alpha * src_f + (1.0f - alpha) * cv::Mat(src_f.size(), src_f.type(), mean);
        cv::min(src_f, 255.0, src_f);
        cv::max(src_f, 0.0,   src_f);
        src_f.convertTo(dst, img.mat().type());
        return Image<Format>(std::move(dst));
    }

private:
    float low_  = 0.8f;
    float high_ = 1.2f;
};

struct ColorJitter : detail::BindMixin<ColorJitter> {
    ColorJitter& brightness(float low, float high) {
        if (low <= 0.0f) throw ParameterError{"brightness.low", "must be > 0", "ColorJitter"};
        if (low > high)  throw ParameterError{"brightness.low", "must be <= high", "ColorJitter"};
        br_low_ = low; br_high_ = high; return *this;
    }
    ColorJitter& contrast(float low, float high) {
        if (low <= 0.0f) throw ParameterError{"contrast.low", "must be > 0", "ColorJitter"};
        if (low > high)  throw ParameterError{"contrast.low", "must be <= high", "ColorJitter"};
        ct_low_ = low; ct_high_ = high; return *this;
    }
    ColorJitter& saturation(float low, float high) {
        if (low <= 0.0f) throw ParameterError{"saturation.low", "must be > 0", "ColorJitter"};
        if (low > high)  throw ParameterError{"saturation.low", "must be <= high", "ColorJitter"};
        sa_low_ = low; sa_high_ = high; return *this;
    }
    ColorJitter& hue(float low, float high) {
        if (std::abs(low) > 180.0f || std::abs(high) > 180.0f)
            throw ParameterError{"hue", "values must be in [-180, 180]", "ColorJitter"};
        if (low > high)
            throw ParameterError{"hue.low", "must be <= high", "ColorJitter"};
        hu_low_ = low; hu_high_ = high; return *this;
    }

    template<BGRFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        std::uniform_real_distribution<float> br_d(br_low_, br_high_);
        std::uniform_real_distribution<float> ct_d(ct_low_, ct_high_);
        std::uniform_real_distribution<float> sa_d(sa_low_, sa_high_);
        std::uniform_real_distribution<float> hu_d(hu_low_, hu_high_);

        float br_f = br_d(rng), ct_f = ct_d(rng);
        float sa_f = sa_d(rng), hu_f = hu_d(rng);

        // Brightness: scale all channels
        cv::Mat bgr_f;
        img.mat().convertTo(bgr_f, CV_32F, 1.0 / 255.0);
        bgr_f *= br_f;
        cv::min(bgr_f, 1.0, bgr_f);
        cv::max(bgr_f, 0.0, bgr_f);

        // Contrast: alpha * img + (1-alpha) * mean
        cv::Scalar mean = cv::mean(bgr_f);
        bgr_f = ct_f * bgr_f + (1.0f - ct_f) * cv::Mat(bgr_f.size(), bgr_f.type(), mean);
        cv::min(bgr_f, 1.0, bgr_f);
        cv::max(bgr_f, 0.0, bgr_f);

        // Saturation & Hue via HSV
        cv::Mat hsv;
        cv::cvtColor(bgr_f, hsv, cv::COLOR_BGR2HSV);  // H:[0,360] S:[0,1] V:[0,1]
        std::vector<cv::Mat> ch;
        cv::split(hsv, ch);
        ch[0] += hu_f;
        ch[0].forEach<float>([](float& v, const int*) {
            v = std::fmod(v, 360.0f);
            if (v < 0.0f) v += 360.0f;
        });
        ch[1] *= sa_f;
        cv::min(ch[1], 1.0, ch[1]);
        cv::max(ch[1], 0.0, ch[1]);
        cv::merge(ch, hsv);
        cv::cvtColor(hsv, bgr_f, cv::COLOR_HSV2BGR);

        static_assert(improc::core::FormatTraits<Format>::cv_type == CV_8UC3,
            "ColorJitter: currently only supports 8-bit BGR images");
        cv::Mat dst;
        bgr_f.convertTo(dst, CV_8UC3, 255.0);
        try {
            return Image<Format>(std::move(dst));
        } catch (const cv::Exception& e) {
            throw AugmentError{"ColorJitter: " + std::string(e.what())};
        }
    }

private:
    float br_low_ = 0.8f,  br_high_ = 1.2f;
    float ct_low_ = 0.8f,  ct_high_ = 1.2f;
    float sa_low_ = 0.8f,  sa_high_ = 1.2f;
    float hu_low_ = -10.f, hu_high_ = 10.0f;
};

struct RandomGrayscale : detail::BindMixin<RandomGrayscale> {
    RandomGrayscale& p(float prob) {
        if (prob < 0.0f || prob > 1.0f)
            throw ParameterError{"p", "must be in [0, 1]", "RandomGrayscale"};
        p_ = prob; return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        std::bernoulli_distribution d(p_);
        if (!d(rng)) return img;
        if constexpr (std::is_same_v<Format, improc::core::BGR>) {
            cv::Mat gray, gray3;
            cv::cvtColor(img.mat(), gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(gray, gray3, cv::COLOR_GRAY2BGR);
            return Image<Format>(std::move(gray3));
        } else {
            return img;
        }
    }

private:
    float p_ = 0.1f;
};

struct RandomSolarize : detail::BindMixin<RandomSolarize> {
    RandomSolarize& threshold(int t) {
        if (t < 0 || t > 255)
            throw ParameterError{"threshold", "must be in [0, 255]", "RandomSolarize"};
        threshold_ = t; return *this;
    }
    RandomSolarize& p(float prob) {
        if (prob < 0.0f || prob > 1.0f)
            throw ParameterError{"p", "must be in [0, 1]", "RandomSolarize"};
        p_ = prob; return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        std::bernoulli_distribution d(p_);
        if (!d(rng)) return img;
        if constexpr (improc::core::FormatTraits<Format>::cv_type == CV_32FC1 ||
                      improc::core::FormatTraits<Format>::cv_type == CV_32FC3) {
            return img;
        } else {
            cv::Mat lut(1, 256, CV_8U);
            auto* p_lut = lut.ptr<uchar>();
            for (int i = 0; i < 256; ++i)
                p_lut[i] = (i >= threshold_) ? static_cast<uchar>(255 - i) : static_cast<uchar>(i);
            cv::Mat dst;
            cv::LUT(img.mat(), lut, dst);
            return Image<Format>(std::move(dst));
        }
    }

private:
    int   threshold_ = 128;
    float p_         = 0.5f;
};

struct RandomPosterize : detail::BindMixin<RandomPosterize> {
    RandomPosterize& bits(int b) {
        if (b < 1 || b > 8)
            throw ParameterError{"bits", "must be in [1, 8]", "RandomPosterize"};
        bits_ = b; return *this;
    }
    RandomPosterize& p(float prob) {
        if (prob < 0.0f || prob > 1.0f)
            throw ParameterError{"p", "must be in [0, 1]", "RandomPosterize"};
        p_ = prob; return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        std::bernoulli_distribution d(p_);
        if (!d(rng)) return img;
        if constexpr (improc::core::FormatTraits<Format>::cv_type == CV_32FC1 ||
                      improc::core::FormatTraits<Format>::cv_type == CV_32FC3) {
            return img;
        } else {
            uchar mask = static_cast<uchar>(0xFF << (8 - bits_)) & 0xFF;
            cv::Mat lut(1, 256, CV_8U);
            auto* p_lut = lut.ptr<uchar>();
            for (int i = 0; i < 256; ++i)
                p_lut[i] = static_cast<uchar>(i) & mask;
            cv::Mat dst;
            cv::LUT(img.mat(), lut, dst);
            return Image<Format>(std::move(dst));
        }
    }

private:
    int   bits_ = 4;
    float p_    = 0.5f;
};

struct RandomEqualize : detail::BindMixin<RandomEqualize> {
    RandomEqualize& p(float prob) {
        if (prob < 0.0f || prob > 1.0f)
            throw ParameterError{"p", "must be in [0, 1]", "RandomEqualize"};
        p_ = prob; return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        std::bernoulli_distribution d(p_);
        if (!d(rng)) return img;
        if constexpr (std::is_same_v<Format, improc::core::BGR>) {
            cv::Mat ycrcb;
            cv::cvtColor(img.mat(), ycrcb, cv::COLOR_BGR2YCrCb);
            std::vector<cv::Mat> ch;
            cv::split(ycrcb, ch);
            cv::equalizeHist(ch[0], ch[0]);
            cv::merge(ch, ycrcb);
            cv::Mat dst;
            cv::cvtColor(ycrcb, dst, cv::COLOR_YCrCb2BGR);
            return Image<Format>(std::move(dst));
        } else if constexpr (std::is_same_v<Format, improc::core::Gray>) {
            cv::Mat dst;
            cv::equalizeHist(img.mat(), dst);
            return Image<Format>(std::move(dst));
        } else {
            return img;
        }
    }

private:
    float p_ = 0.5f;
};

} // namespace improc::ml
