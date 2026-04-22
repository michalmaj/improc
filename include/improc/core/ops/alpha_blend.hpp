// include/improc/core/ops/alpha_blend.hpp
#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/format_traits.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Alpha composite: blends a BGRA overlay onto a BGR background.
 *
 * Per-pixel: `out = bg * (1 - a/255) + overlay_rgb * (a/255)`
 * where `a` is the overlay's alpha channel.
 *
 * Background and overlay must have identical spatial dimensions.
 *
 * @throws ParameterError if sizes differ.
 *
 * @code
 * Image<BGR> result = background | AlphaBlend{bgra_overlay};
 * @endcode
 */
struct AlphaBlend {
    explicit AlphaBlend(Image<BGRA> overlay) : overlay_(std::move(overlay)) {}

    Image<BGR> operator()(Image<BGR> img) const {
        if (img.rows() != overlay_.rows() || img.cols() != overlay_.cols())
            throw ParameterError{"overlay", "size must match background", "AlphaBlend"};

        std::vector<cv::Mat> planes;
        cv::split(overlay_.mat(), planes); // planes: B, G, R, A

        cv::Mat alpha_f;
        planes[3].convertTo(alpha_f, CV_32F, 1.0 / 255.0);

        // Expand single-channel alpha and inv-alpha to 3 channels using merge (safe for CV_32F)
        cv::Mat inv_alpha_f;
        cv::subtract(cv::Scalar(1.0f), alpha_f, inv_alpha_f);

        cv::Mat alpha_3c, inv_alpha_3c;
        cv::merge(std::vector<cv::Mat>{alpha_f, alpha_f, alpha_f}, alpha_3c);
        cv::merge(std::vector<cv::Mat>{inv_alpha_f, inv_alpha_f, inv_alpha_f}, inv_alpha_3c);

        cv::Mat bg_f, ol_f;
        img.mat().convertTo(bg_f, CV_32FC3);

        cv::Mat ol_bgr;
        cv::merge(std::vector<cv::Mat>{planes[0], planes[1], planes[2]}, ol_bgr);
        ol_bgr.convertTo(ol_f, CV_32FC3);

        cv::Mat bg_weighted, ol_weighted, result_f;
        cv::multiply(bg_f, inv_alpha_3c, bg_weighted);
        cv::multiply(ol_f, alpha_3c,     ol_weighted);
        cv::add(bg_weighted, ol_weighted, result_f);

        cv::Mat dst;
        result_f.convertTo(dst, CV_8UC3);
        return Image<BGR>(std::move(dst));
    }

private:
    Image<BGRA> overlay_;
};

} // namespace improc::core
