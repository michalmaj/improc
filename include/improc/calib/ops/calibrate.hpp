#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include "improc/calib/ops/calib_types.hpp"
#include "improc/exceptions.hpp"

namespace improc::calib {

/// @brief Generates 3-D object points for a flat chessboard (Z = 0 plane).
/// @param board_size Inner corner count; `board_size.width × board_size.height` points are returned.
/// @param square_size Physical side length of one square in any consistent unit.
/// @return `board_size.width * board_size.height` points in row-major order.
/// @throws improc::ParameterError if any dimension of `board_size` <= 0 or `square_size` <= 0.
inline std::vector<cv::Point3f> make_chessboard_points(cv::Size board_size,
                                                        float square_size) {
    if (board_size.width <= 0 || board_size.height <= 0)
        throw improc::ParameterError{"board_size", "dimensions must be positive", "make_chessboard_points"};
    if (square_size <= 0.f)
        throw improc::ParameterError{"square_size", "must be positive", "make_chessboard_points"};
    std::vector<cv::Point3f> pts;
    pts.reserve(board_size.width * board_size.height);
    for (int r = 0; r < board_size.height; ++r)
        for (int c = 0; c < board_size.width; ++c)
            pts.push_back({c * square_size, r * square_size, 0.f});
    return pts;
}

/**
 * @brief Calibrates a single camera from a set of chessboard views.
 *
 * @code
 * auto pts3d = make_chessboard_points({9, 6}, 0.025f);
 * std::vector obj_pts(views.size(), pts3d);
 * auto result = CalibrateCamera()(obj_pts, img_pts, image_size);
 * @endcode
 */
struct CalibrateCamera {
    /// @brief Sets OpenCV calibration flags (default: 0).
    CalibrateCamera& flags(int f) { flags_ = f; return *this; }

    /// @brief Runs the calibration.
    /// @throws improc::ParameterError if sizes mismatch or fewer than 3 views are provided.
    CalibrationResult operator()(const std::vector<std::vector<cv::Point3f>>& obj_pts,
                                 const std::vector<std::vector<cv::Point2f>>& img_pts,
                                 cv::Size image_size) const;

private:
    int flags_ = 0;
};

} // namespace improc::calib
