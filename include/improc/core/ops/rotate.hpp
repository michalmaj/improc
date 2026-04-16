#pragma once

#include <optional>
#include <stdexcept>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"

namespace improc::core {

struct Rotate {
    Rotate& angle(double deg) { angle_ = deg; return *this; }
    Rotate& scale(double s) {
        if (s <= 0.0) throw std::invalid_argument("Rotate: scale must be positive");
        scale_ = s;
        return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        if (!angle_) {
            throw std::invalid_argument("Rotate: angle must be set");
        }
        cv::Point2f center(img.cols() / 2.0f, img.rows() / 2.0f);
        cv::Mat M = cv::getRotationMatrix2D(center, *angle_, scale_);
        cv::Mat dst;
        cv::warpAffine(img.mat(), dst, M, cv::Size(img.cols(), img.rows()));
        return Image<Format>(std::move(dst));
    }

private:
    std::optional<double> angle_;
    double scale_ = 1.0;
};

} // namespace improc::core
