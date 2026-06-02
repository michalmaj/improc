#include "improc/calib/ops/calibrate.hpp"
#include <opencv2/calib3d.hpp>

namespace improc::calib {

CalibrationResult CalibrateCamera::operator()(
        const std::vector<std::vector<cv::Point3f>>& obj_pts,
        const std::vector<std::vector<cv::Point2f>>& img_pts,
        cv::Size image_size) const {
    if (obj_pts.size() != img_pts.size())
        throw improc::ParameterError{"obj_pts", "must have the same number of views as img_pts", "CalibrateCamera"};
    if (obj_pts.size() < 3)
        throw improc::ParameterError{"obj_pts", "at least 3 views required", "CalibrateCamera"};
    if (image_size.width <= 0 || image_size.height <= 0)
        throw improc::ParameterError{"image_size", "dimensions must be positive", "CalibrateCamera"};

    CalibrationResult result;
    result.rms = cv::calibrateCamera(obj_pts, img_pts, image_size,
                                     result.camera_matrix, result.dist_coeffs,
                                     result.rvecs, result.tvecs,
                                     flags_);
    return result;
}

} // namespace improc::calib
