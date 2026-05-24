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

struct ArucoDict {
    cv::aruco::Dictionary operator()(cv::aruco::PredefinedDictionaryType type) const;
};

// ── DetectAruco ───────────────────────────────────────────────────────────────

struct DetectAruco {
    ArucoResult operator()(Image<BGR>  img, const cv::aruco::Dictionary& dict) const;
    ArucoResult operator()(Image<Gray> img, const cv::aruco::Dictionary& dict) const;
};

// ── DrawAruco ─────────────────────────────────────────────────────────────────

struct DrawAruco {
    /// Sets the axis length used when drawing 3D axes (world units). Default: 0.05.
    DrawAruco& axis_length(float l) { axis_length_ = l; return *this; }

    // Overload 1: draws marker outlines + ID numbers only.
    cv::Mat operator()(cv::Mat img, const ArucoResult& result) const;

    // Overload 2: draws marker outlines + IDs + coordinate axes per marker.
    // Uses cv::drawFrameAxes. Iterates poses and draws one axis frame per marker.
    cv::Mat operator()(cv::Mat img,
                       const ArucoResult& result,
                       const std::vector<ArucoPoseResult>& poses,
                       const cv::Mat& K,
                       const cv::Mat& dist) const;

private:
    float axis_length_ = 0.05f;
};

// ── GenerateAruco ─────────────────────────────────────────────────────────────

struct GenerateAruco {
    /// Sets the number of border bits. Default: 1.
    GenerateAruco& border_bits(int b) { border_bits_ = b; return *this; }

    // Returns a square Gray image of size side_pixels × side_pixels.
    // Throws std::invalid_argument if id < 0 or side_pixels < 1.
    Image<Gray> operator()(const cv::aruco::Dictionary& dict,
                           int id,
                           int side_pixels) const;

private:
    int border_bits_ = 1;
};

// ── ArucoPose ─────────────────────────────────────────────────────────────────

struct ArucoPose {
    // Estimates pose per detected marker via cv::solvePnP (not deprecated estimatePoseSingleMarkers).
    // Object points per marker (half = marker_length/2):
    //   top-left=(-half, half, 0), top-right=(half, half, 0),
    //   bottom-right=(half, -half, 0), bottom-left=(-half, -half, 0)
    // Returns one ArucoPoseResult per detected marker (same order as ArucoResult).
    std::vector<ArucoPoseResult> operator()(const ArucoResult& result,
                                            const cv::Mat& K,
                                            const cv::Mat& dist,
                                            float marker_length) const;
};

// ── CharucoBoard ──────────────────────────────────────────────────────────────

struct CharucoBoard {
    // board_size: number of squares (cols × rows), e.g. {5, 7} = 5 columns, 7 rows.
    CharucoBoard& board_size(cv::Size s) { board_size_    = s; has_size_ = true; return *this; }
    CharucoBoard& square_length(float s) { square_length_ = s; return *this; }
    CharucoBoard& marker_length(float m) { marker_length_ = m; return *this; }

    // Throws std::invalid_argument if board_size not set, square_length <= 0, or marker_length <= 0.
    // Overload 1: basic detection, no subpixel refinement.
    CharucoResult operator()(Image<BGR> img, const cv::aruco::Dictionary& dict) const;
    // Overload 2: enables subpixel corner refinement via CharucoParameters.
    CharucoResult operator()(Image<BGR> img,
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
