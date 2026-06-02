// include/improc/calib/ops/aruco.hpp
#pragma once
#include <vector>
#include <stdexcept>
#include <opencv2/core.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>  // cv::aruco::Dictionary, ArucoDetector, PredefinedDictionaryType (includes aruco_board.hpp)
#include "improc/calib/ops/calib_types.hpp"
#include "improc/core/image.hpp"

namespace improc::calib {

using improc::core::Image;
using improc::core::BGR;
using improc::core::Gray;

// ── ArucoDict ─────────────────────────────────────────────────────────────────

/**
 * @brief Retrieves a predefined ArUco dictionary by type.
 *
 * @code
 * auto dict = ArucoDict()(cv::aruco::DICT_4X4_50);
 * @endcode
 */
struct ArucoDict {
    [[nodiscard]] cv::aruco::Dictionary operator()(cv::aruco::PredefinedDictionaryType type) const;
};

// ── DetectAruco ───────────────────────────────────────────────────────────────

/**
 * @brief Detects ArUco markers in a BGR or Gray image.
 *
 * @return `ArucoResult` containing corners, IDs, and rejected candidates.
 */
struct DetectAruco {
    [[nodiscard]] ArucoResult operator()(Image<BGR>  img, const cv::aruco::Dictionary& dict) const;
    [[nodiscard]] ArucoResult operator()(Image<Gray> img, const cv::aruco::Dictionary& dict) const;
};

// ── DrawAruco ─────────────────────────────────────────────────────────────────

/**
 * @brief Draws ArUco marker outlines, IDs, and optionally 3-D pose axes onto an image.
 */
struct DrawAruco {
    /// @brief Sets the axis length used when drawing 3-D frames (default: 0.05 world units).
    DrawAruco& axis_length(float l) { axis_length_ = l; return *this; }

    // Overload 1: draws marker outlines + ID numbers only.
    [[nodiscard]] cv::Mat operator()(cv::Mat img, const ArucoResult& result) const;

    // Overload 2: draws marker outlines + IDs + coordinate axes per marker.
    // Uses cv::drawFrameAxes. Iterates poses and draws one axis frame per marker.
    [[nodiscard]] cv::Mat operator()(cv::Mat img,
                       const ArucoResult& result,
                       const std::vector<ArucoPoseResult>& poses,
                       const cv::Mat& K,
                       const cv::Mat& dist) const;

private:
    float axis_length_ = 0.05f;
};

// ── GenerateAruco ─────────────────────────────────────────────────────────────

/**
 * @brief Renders a single ArUco marker as a square grayscale image.
 */
struct GenerateAruco {
    /// @brief Sets the number of border bits (default: 1).
    GenerateAruco& border_bits(int b) { border_bits_ = b; return *this; }

    /// @brief Renders marker `id` from `dict` at `side_pixels × side_pixels`.
    /// @throws improc::ParameterError if `id` < 0 or `side_pixels` < 1.
    [[nodiscard]] Image<Gray> operator()(const cv::aruco::Dictionary& dict,
                           int id,
                           int side_pixels) const;

private:
    int border_bits_ = 1;
};

// ── ArucoPose ─────────────────────────────────────────────────────────────────

/**
 * @brief Estimates the 6-DoF pose of each detected ArUco marker via `cv::solvePnP`.
 *
 * Object points are defined with the marker centre at the origin:
 * top-left = (-half, half, 0), top-right = (half, half, 0),
 * bottom-right = (half, -half, 0), bottom-left = (-half, -half, 0).
 */
struct ArucoPose {
    /// @brief Estimates poses for all markers in `result`.
    /// @param result       Detected markers from `DetectAruco`.
    /// @param K            3×3 camera matrix.
    /// @param dist         Distortion coefficients.
    /// @param marker_length Physical marker side length in world units.
    /// @return One `ArucoPoseResult` per detected marker, in the same order as `result`.
    [[nodiscard]] std::vector<ArucoPoseResult> operator()(const ArucoResult& result,
                                            const cv::Mat& K,
                                            const cv::Mat& dist,
                                            float marker_length) const;
};

// ── CharucoBoard ──────────────────────────────────────────────────────────────

/**
 * @brief Detects ChArUco board corners with optional subpixel refinement.
 *
 * @code
 * auto result = CharucoBoard()
 *     .board_size({5, 7})
 *     .square_length(0.04f)
 *     .marker_length(0.02f)
 *     (img, dict);
 * @endcode
 */
struct CharucoBoard {
    /// @brief Sets the board dimensions as (cols, rows) of chessboard squares.
    CharucoBoard& board_size(cv::Size s) { board_size_    = s; has_size_ = true; return *this; }
    /// @brief Sets the physical square side length in world units (e.g. metres).
    CharucoBoard& square_length(float s) { square_length_ = s; return *this; }
    /// @brief Sets the physical ArUco marker side length in world units.
    CharucoBoard& marker_length(float m) { marker_length_ = m; return *this; }

    /// @brief Detects board corners (no subpixel refinement).
    /// @throws improc::ParameterError if `board_size` not set, or lengths <= 0.
    [[nodiscard]] CharucoResult operator()(Image<BGR> img, const cv::aruco::Dictionary& dict) const;
    /// @brief Detects board corners with subpixel refinement using camera intrinsics.
    [[nodiscard]] CharucoResult operator()(Image<BGR> img,
                             const cv::aruco::Dictionary& dict,
                             const cv::Mat& K,
                             const cv::Mat& dist) const;

private:
    cv::Size board_size_{};
    float    square_length_ = 0.f;
    float    marker_length_ = 0.f;
    bool     has_size_      = false;
};

} // namespace improc::calib
