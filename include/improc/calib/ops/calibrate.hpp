#pragma once
#include <stdexcept>
#include <vector>
#include <opencv2/calib3d.hpp>
#include "improc/calib/ops/calib_types.hpp"

namespace improc::calib {

// Generates object points for one flat chessboard view (Z=0 plane).
// board_size = inner corners; square_size in any consistent unit.
// Returns board_size.width × board_size.height points.
inline std::vector<cv::Point3f> make_chessboard_points(cv::Size board_size,
                                                        float square_size) {
    std::vector<cv::Point3f> pts;
    pts.reserve(board_size.width * board_size.height);
    for (int r = 0; r < board_size.height; ++r)
        for (int c = 0; c < board_size.width; ++c)
            pts.push_back({c * square_size, r * square_size, 0.f});
    return pts;
}

struct CalibrateCamera {
    CalibrateCamera& flags(int f) { flags_ = f; return *this; }

    // Throws std::invalid_argument if sizes mismatch or fewer than 3 views.
    CalibrationResult operator()(const std::vector<std::vector<cv::Point3f>>& obj_pts,
                                 const std::vector<std::vector<cv::Point2f>>& img_pts,
                                 cv::Size image_size) const;

private:
    int flags_ = 0;
};

} // namespace improc::calib
