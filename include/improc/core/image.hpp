// include/improc/core/image.hpp
#pragma once

#include <opencv2/core.hpp>
#include <stdexcept>
#include <format>
#include "improc/core/format_traits.hpp"

namespace improc::core {

template<typename Format>
class Image {
public:
    explicit Image(cv::Mat mat) : mat_(std::move(mat)) {
        if (mat_.empty()) {
            throw std::invalid_argument("Image: mat must not be empty");
        }
        if (mat_.type() != FormatTraits<Format>::cv_type) {
            throw std::invalid_argument(
                std::format("Image: expected cv_type {}, got {}",
                    FormatTraits<Format>::cv_type, mat_.type()));
        }
    }

    Image clone() const { return Image(mat_.clone()); }

    [[nodiscard]] cv::Mat&       mat()       { return mat_; }
    [[nodiscard]] const cv::Mat& mat() const { return mat_; }

    [[nodiscard]] int  rows()  const { return mat_.rows; }
    [[nodiscard]] int  cols()  const { return mat_.cols; }
    [[nodiscard]] bool empty() const { return mat_.empty(); }

    Image(const Image&)            = default;
    Image& operator=(const Image&) = default;
    Image(Image&&)                 = default;
    Image& operator=(Image&&)      = default;

private:
    cv::Mat mat_;
};

} // namespace improc::core
