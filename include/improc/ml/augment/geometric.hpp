// include/improc/ml/augment/geometric.hpp
#pragma once

#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/core/ops/axis.hpp"
#include "improc/ml/augment/detail.hpp"

namespace improc::ml {

using improc::core::AnyFormat;
using improc::core::Image;

struct RandomFlip : detail::BindMixin<RandomFlip> {
    RandomFlip& p(float prob) {
        if (prob < 0.0f || prob > 1.0f)
            throw std::invalid_argument("RandomFlip: p must be in [0, 1]");
        p_ = prob; return *this;
    }
    RandomFlip& axis(core::Axis a) { axis_ = a; return *this; }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        std::bernoulli_distribution d(p_);
        if (!d(rng)) return img;
        int flip_code;
        switch (axis_) {
            case core::Axis::Horizontal: flip_code =  1; break;
            case core::Axis::Vertical:   flip_code =  0; break;
            case core::Axis::Both:       flip_code = -1; break;
            default: std::unreachable();
        }
        cv::Mat dst;
        cv::flip(img.mat(), dst, flip_code);
        return Image<Format>(std::move(dst));
    }

private:
    float      p_    = 0.5f;
    core::Axis axis_ = core::Axis::Horizontal;
};

struct RandomRotate : detail::BindMixin<RandomRotate> {
    RandomRotate& range(float min_deg, float max_deg) {
        if (min_deg > max_deg)
            throw std::invalid_argument("RandomRotate: min_deg must be <= max_deg");
        min_deg_ = min_deg; max_deg_ = max_deg; return *this;
    }
    RandomRotate& scale(float s) {
        if (s <= 0.0f)
            throw std::invalid_argument("RandomRotate: scale must be > 0");
        scale_ = s; return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        std::uniform_real_distribution<float> d(min_deg_, max_deg_);
        float angle = d(rng);
        cv::Point2f center(img.cols() / 2.0f, img.rows() / 2.0f);
        cv::Mat M = cv::getRotationMatrix2D(center, angle, scale_);
        cv::Mat dst;
        try {
            cv::warpAffine(img.mat(), dst, M, img.mat().size());
        } catch (const cv::Exception& e) {
            throw std::runtime_error("RandomRotate: " + std::string(e.what()));
        }
        return Image<Format>(std::move(dst));
    }

private:
    float min_deg_ = -15.0f;
    float max_deg_ =  15.0f;
    float scale_   =   1.0f;
};

struct RandomCrop : detail::BindMixin<RandomCrop> {
    RandomCrop& width(int w) {
        if (w <= 0) throw std::invalid_argument("RandomCrop: width must be positive");
        width_ = w; return *this;
    }
    RandomCrop& height(int h) {
        if (h <= 0) throw std::invalid_argument("RandomCrop: height must be positive");
        height_ = h; return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        if (width_ <= 0 || height_ <= 0)
            throw std::invalid_argument("RandomCrop: width and height must both be set");
        if (width_ > img.cols() || height_ > img.rows())
            throw std::invalid_argument("RandomCrop: crop size exceeds image dimensions");
        std::uniform_int_distribution<int> x_d(0, img.cols() - width_);
        std::uniform_int_distribution<int> y_d(0, img.rows() - height_);
        int x = x_d(rng);
        int y = y_d(rng);
        cv::Mat dst = img.mat()(cv::Rect(x, y, width_, height_)).clone();
        return Image<Format>(std::move(dst));
    }

private:
    int width_  = 0;
    int height_ = 0;
};

struct RandomResize : detail::BindMixin<RandomResize> {
    RandomResize& range(int min_side, int max_side) {
        if (min_side <= 0)
            throw std::invalid_argument("RandomResize: min_side must be > 0");
        if (min_side > max_side)
            throw std::invalid_argument("RandomResize: min_side must be <= max_side");
        min_side_ = min_side; max_side_ = max_side; return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        std::uniform_int_distribution<int> d(min_side_, max_side_);
        int target = d(rng);
        const int h = img.rows();
        const int w = img.cols();
        int new_w, new_h;
        if (h <= w) {
            new_h = target;
            new_w = static_cast<int>(std::round(static_cast<double>(w) * target / h));
        } else {
            new_w = target;
            new_h = static_cast<int>(std::round(static_cast<double>(h) * target / w));
        }
        cv::Mat dst;
        cv::resize(img.mat(), dst, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
        return Image<Format>(std::move(dst));
    }

private:
    int min_side_ = 224;
    int max_side_ = 256;
};

} // namespace improc::ml
