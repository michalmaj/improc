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

/**
 * @brief Detects inner chessboard corners using the more robust SB algorithm.
 *
 * Uses `cv::findChessboardCornersSB` which provides sub-pixel accuracy without
 * a separate refinement step. `board_size` must be set before invoking.
 * BGR input is auto-converted to Gray internally.
 *
 * @throws std::invalid_argument if board_size has not been set.
 *
 * @code
 * auto result = gray_img | FindChessboardCornersSB{}.board_size({9, 6});
 * if (result.found) { ... result.corners ... }
 * @endcode
 */
struct FindChessboardCornersSB {
    /// @brief Sets the inner corner count (cols × rows). Required before use.
    FindChessboardCornersSB& board_size(cv::Size s) { board_size_ = s; has_board_size_ = true; return *this; }
    /// @brief Overrides the default detection flags.
    FindChessboardCornersSB& flags(int f)            { flags_ = f; return *this; }

    /// @brief Finds chessboard corners in a grayscale image.
    FindChessboardResult operator()(Image<Gray> img) const;
    /// @brief Finds chessboard corners in a BGR image (auto-converted to Gray).
    FindChessboardResult operator()(Image<BGR>  img) const;

private:
    cv::Size board_size_{};
    bool     has_board_size_ = false;
    int      flags_ = 0;
};

/**
 * @brief Refines detected chessboard corners to sub-pixel accuracy.
 *
 * Wraps `cv::cornerSubPix`. Takes a Gray image and a vector of initial corner
 * estimates, returns a new vector with refined positions.
 *
 * @code
 * auto refined = RefineCorners{}.win_size(11)(gray_img, found.corners);
 * @endcode
 */
struct RefineCorners {
    /// @brief Sets the half-size of the search window (default: 11).
    RefineCorners& win_size(int s)   { win_size_  = {s, s}; return *this; }
    /// @brief Sets the half-size of the dead zone (default: {-1,-1} = none).
    RefineCorners& zero_zone(int z)  { zero_zone_ = {z, z}; return *this; }
    /// @brief Sets the maximum number of iterations (default: 30).
    RefineCorners& max_iter(int n)   { max_iter_  = n;       return *this; }
    /// @brief Sets the convergence epsilon (default: 0.001).
    RefineCorners& epsilon(double e) { epsilon_   = e;       return *this; }

    /// @brief Refines corners in a grayscale image. Returns refined corner positions.
    std::vector<cv::Point2f> operator()(Image<Gray>               img,
                                        std::vector<cv::Point2f>  corners) const;

private:
    cv::Size win_size_  = {11, 11};
    cv::Size zero_zone_ = {-1, -1};
    int      max_iter_  = 30;
    double   epsilon_   = 0.001;
};

} // namespace improc::calib
