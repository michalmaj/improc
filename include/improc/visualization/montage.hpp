// include/improc/visualization/montage.hpp
#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::visualization {

using improc::core::Image;
using improc::core::BGR;

/**
 * @brief Arranges a collection of BGR images into a grid (montage).
 *
 * Useful for debugging ML pipelines: pass a batch of images and get back a
 * single `Image<BGR>` showing them side-by-side. Images are resized to
 * `cell_size`; empty cells are filled with `background` color.
 *
 * @throws improc::ParameterError if the image vector is empty.
 *
 * @code
 * Image<BGR> grid = Montage{batch}
 *     .cols(4)
 *     .cell_size(224, 224)
 *     .gap(4)();
 * @endcode
 */
struct Montage {
    /**
     * @brief Constructs the montage op with the given images.
     * @throws improc::ParameterError if `images` is empty.
     */
    explicit Montage(std::vector<Image<BGR>> images);

    /// @brief Sets number of grid columns. Default: `ceil(sqrt(n))`.
    /// @throws improc::ParameterError if `c` <= 0.
    Montage& cols(int c);

    /// @brief Sets each cell's pixel dimensions. Default: size of first image.
    /// @throws improc::ParameterError if `w` or `h` <= 0.
    Montage& cell_size(int w, int h);

    /// @brief Sets pixel gap between cells. Default: 0.
    /// @throws improc::ParameterError if `g` < 0.
    Montage& gap(int g);

    /// @brief Sets background fill color for gaps and empty cells. Default: black.
    Montage& background(cv::Scalar color);

    /// @brief Renders and returns the grid as `Image<BGR>`.
    Image<BGR> operator()() const;

private:
    std::vector<Image<BGR>> images_;
    int        cols_  = 0;
    int        cell_w_= 0;
    int        cell_h_= 0;
    int        gap_   = 0;
    cv::Scalar bg_    = {0, 0, 0};
};

} // namespace improc::visualization
