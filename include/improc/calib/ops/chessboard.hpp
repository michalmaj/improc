// include/improc/calib/ops/chessboard.hpp
#pragma once
#include <stdexcept>
#include <opencv2/calib3d.hpp>
#include "improc/core/image.hpp"
#include "improc/calib/ops/calib_types.hpp"

namespace improc::calib {

using improc::core::Image;
using improc::core::Gray;
using improc::core::BGR;

/**
 * @brief Detects inner chessboard corners in a Gray or BGR image.
 *
 * `board_size` must be set before invoking the operator — it specifies the
 * number of inner corners per row and column (e.g., `{9, 6}` for a 10×7
 * square board). BGR input is auto-converted to Gray internally.
 *
 * @throws std::invalid_argument if board_size has not been set.
 *
 * @code
 * auto result = gray_img | FindChessboardCorners{}.board_size({9, 6});
 * if (result.found) { ... result.corners ... }
 * @endcode
 */
struct FindChessboardCorners {
    /// @brief Sets the inner corner count (cols × rows). Required before use.
    FindChessboardCorners& board_size(cv::Size s) { board_size_ = s; has_board_size_ = true; return *this; }
    /// @brief Overrides the default detection flags (ADAPTIVE_THRESH | NORMALIZE_IMAGE).
    FindChessboardCorners& flags(int f)            { flags_ = f; return *this; }

    /// @brief Finds chessboard corners in a grayscale image.
    FindChessboardResult operator()(Image<Gray> img) const;
    /// @brief Finds chessboard corners in a BGR image (auto-converted to Gray).
    FindChessboardResult operator()(Image<BGR>  img) const;

private:
    cv::Size board_size_{};
    bool     has_board_size_ = false;
    int      flags_ = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
};

} // namespace improc::calib
