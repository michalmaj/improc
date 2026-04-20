// include/improc/core/ops/apply_mask.hpp
#pragma once

#include <optional>
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

struct ApplyMask {
    ApplyMask& mask(Image<Gray> m) {
        mask_ = std::move(m);
        return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        if (!mask_)
            throw ParameterError{"mask", "must be set before calling operator()", "ApplyMask"};
        if (mask_->rows() != img.rows() || mask_->cols() != img.cols())
            throw ParameterError{"mask", "must have the same dimensions as the source image", "ApplyMask"};
        cv::Mat dst = cv::Mat::zeros(img.mat().size(), img.mat().type());
        img.mat().copyTo(dst, mask_->mat());
        return Image<Format>(std::move(dst));
    }

private:
    std::optional<Image<Gray>> mask_;
};

} // namespace improc::core
