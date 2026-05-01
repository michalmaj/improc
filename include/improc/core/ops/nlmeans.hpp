// include/improc/core/ops/nlmeans.hpp
#pragma once

#include <format>
#include <opencv2/photo.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Non-Local Means denoising — reduces noise by averaging similar patches.
 *
 * On `Image<Gray>`: delegates to `cv::fastNlMeansDenoising`.
 * On `Image<BGR>`:  delegates to `cv::fastNlMeansDenoisingColored`.
 *
 * @throws improc::ParameterError if h or h_color are not positive.
 * @throws improc::ParameterError if template_window_size or search_window_size
 *         are not odd and positive.
 *
 * @code
 * Image<Gray> denoised = noisy | NLMeansDenoising{}.h(10.0f);
 * Image<BGR>  denoised = noisy | NLMeansDenoising{}.h(10.0f).h_color(10.0f);
 * @endcode
 */
struct NLMeansDenoising {
    /// @brief Filter strength for luminance. Default 3.0. Must be > 0.
    NLMeansDenoising& h(float v) {
        if (v <= 0.0f)
            throw ParameterError{"h", "must be positive", "NLMeansDenoising"};
        h_ = v; return *this;
    }
    /// @brief Filter strength for color components (BGR only). Default 3.0. Must be > 0.
    NLMeansDenoising& h_color(float v) {
        if (v <= 0.0f)
            throw ParameterError{"h_color", "must be positive", "NLMeansDenoising"};
        h_color_ = v; return *this;
    }
    /// @brief Patch size for template matching. Default 7. Must be odd and positive.
    NLMeansDenoising& template_window_size(int v) {
        if (v <= 0 || v % 2 == 0)
            throw ParameterError{"template_window_size",
                std::format("must be odd and positive, got {}", v), "NLMeansDenoising"};
        template_window_size_ = v; return *this;
    }
    /// @brief Size of the search area. Default 21. Must be odd and positive.
    NLMeansDenoising& search_window_size(int v) {
        if (v <= 0 || v % 2 == 0)
            throw ParameterError{"search_window_size",
                std::format("must be odd and positive, got {}", v), "NLMeansDenoising"};
        search_window_size_ = v; return *this;
    }

    /// @brief Denoises a single-channel gray image.
    Image<Gray> operator()(Image<Gray> img) const;
    /// @brief Denoises a BGR color image.
    Image<BGR>  operator()(Image<BGR>  img) const;

private:
    float h_                    = 3.0f;
    float h_color_              = 3.0f;
    int   template_window_size_ = 7;
    int   search_window_size_   = 21;
};

} // namespace improc::core
