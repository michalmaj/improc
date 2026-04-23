// include/improc/visualization/show.hpp
#pragma once

#include <string>
#include <opencv2/highgui.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::visualization {

using improc::core::Image;
using improc::core::BGR;

/**
 * @brief Pipeline passthrough op that displays the image in a named window.
 *
 * Returns its input unchanged, enabling composition: `img | Show{"win"} | next_op`.
 * `wait_ms(0)` blocks until a key is pressed; `wait_ms(N)` waits N milliseconds.
 *
 * @code
 * img | Show{"preview"}.wait_ms(1) | ToGray{};
 * @endcode
 */
struct Show {
    /// @brief Constructs the display op with the given OpenCV window name.
    explicit Show(std::string window_name)
        : window_name_(std::move(window_name)) {}

    /// @brief Sets the `cv::waitKey` delay in milliseconds (0 = block until key press).
    /// @throws improc::ParameterError if `ms` < 0.
    Show& wait_ms(int ms) {
        if (ms < 0) throw ParameterError{"wait_ms", "must be >= 0", "Show"};
        wait_ms_ = ms;
        return *this;
    }

    /// @brief Displays the image via `cv::imshow`, waits `wait_ms` ms, and returns the image unchanged.
    ///
    /// Accepts only `Image<BGR>`; convert `Gray`/`Float32` images before passing.
    Image<BGR> operator()(Image<BGR> img) const {
        cv::imshow(window_name_, img.mat());
        cv::waitKey(wait_ms_);
        return img;
    }

private:
    std::string window_name_;
    int         wait_ms_ = 0;
};

} // namespace improc::visualization
