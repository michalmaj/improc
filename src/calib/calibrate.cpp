#include "improc/calib/ops/calibrate.hpp"

namespace improc::calib {

CalibrationResult CalibrateCamera::operator()(
        const std::vector<std::vector<cv::Point3f>>& obj_pts,
        const std::vector<std::vector<cv::Point2f>>& img_pts,
        cv::Size image_size) const {
    if (obj_pts.size() != img_pts.size())
        throw std::invalid_argument(
            "CalibrateCamera: obj_pts and img_pts must have the same number of views");
    if (obj_pts.size() < 3)
        throw std::invalid_argument(
            "CalibrateCamera: at least 3 views required");

    CalibrationResult result;
    result.rms = cv::calibrateCamera(obj_pts, img_pts, image_size,
                                     result.camera_matrix, result.dist_coeffs,
                                     result.rvecs, result.tvecs,
                                     flags_);
    return result;
}

} // namespace improc::calib
