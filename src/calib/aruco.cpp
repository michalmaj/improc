// src/calib/aruco.cpp
#include "improc/calib/ops/aruco.hpp"
#include <opencv2/objdetect/charuco_detector.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace improc::calib {

cv::aruco::Dictionary ArucoDict::operator()(cv::aruco::PredefinedDictionaryType type) const {
    return cv::aruco::getPredefinedDictionary(type);
}

ArucoResult DetectAruco::operator()(Image<BGR> img,
                                     const cv::aruco::Dictionary& dict) const {
    cv::aruco::ArucoDetector detector(dict);
    ArucoResult result;
    detector.detectMarkers(img.mat(), result.corners, result.ids, result.rejected);
    return result;
}

ArucoResult DetectAruco::operator()(Image<Gray> img,
                                     const cv::aruco::Dictionary& dict) const {
    cv::aruco::ArucoDetector detector(dict);
    ArucoResult result;
    detector.detectMarkers(img.mat(), result.corners, result.ids, result.rejected);
    return result;
}

cv::Mat DrawAruco::operator()(cv::Mat img, const ArucoResult& result) const {
    cv::Mat out = img.clone();
    if (!result.ids.empty())
        cv::aruco::drawDetectedMarkers(out, result.corners, result.ids);
    return out;
}

cv::Mat DrawAruco::operator()(cv::Mat img,
                               const ArucoResult& result,
                               const std::vector<ArucoPoseResult>& poses,
                               const cv::Mat& K,
                               const cv::Mat& dist) const {
    cv::Mat out = operator()(img, result);
    for (const auto& pose : poses)
        cv::drawFrameAxes(out, K, dist, pose.rvec, pose.tvec, axis_length_);
    return out;
}

Image<Gray> GenerateAruco::operator()(const cv::aruco::Dictionary& dict,
                                       int id, int side_pixels) const {
    if (id < 0)
        throw std::invalid_argument(
            "GenerateAruco: id must be >= 0");
    if (side_pixels < 1)
        throw std::invalid_argument(
            "GenerateAruco: side_pixels must be >= 1");
    cv::Mat img;
    cv::aruco::generateImageMarker(dict, id, side_pixels, img, border_bits_);
    return Image<Gray>(img);
}

std::vector<ArucoPoseResult> ArucoPose::operator()(const ArucoResult& result,
                                                    const cv::Mat& K,
                                                    const cv::Mat& dist,
                                                    float marker_length) const {
    const float half = marker_length / 2.f;
    const std::vector<cv::Point3f> obj_pts = {
        {-half,  half, 0.f},  // top-left
        { half,  half, 0.f},  // top-right
        { half, -half, 0.f},  // bottom-right
        {-half, -half, 0.f},  // bottom-left
    };
    std::vector<ArucoPoseResult> poses;
    poses.reserve(result.corners.size());
    for (size_t i = 0; i < result.corners.size(); ++i) {
        ArucoPoseResult pr;
        pr.id = result.ids[i];
        cv::solvePnP(obj_pts, result.corners[i], K, dist, pr.rvec, pr.tvec);
        poses.push_back(std::move(pr));
    }
    return poses;
}

namespace {
void validate_charuco(bool has_size, float sq, float mk) {
    if (!has_size)
        throw std::invalid_argument(
            "CharucoBoard: board_size must be set before use");
    if (sq <= 0.f)
        throw std::invalid_argument(
            "CharucoBoard: square_length must be positive");
    if (mk <= 0.f)
        throw std::invalid_argument(
            "CharucoBoard: marker_length must be positive");
}
} // namespace

CharucoResult CharucoBoard::operator()(Image<BGR>, const cv::aruco::Dictionary&) const {
    validate_charuco(has_size_, square_length_, marker_length_);
    return {};
}

CharucoResult CharucoBoard::operator()(Image<BGR>, const cv::aruco::Dictionary&,
                                        const cv::Mat&, const cv::Mat&) const {
    validate_charuco(has_size_, square_length_, marker_length_);
    return {};
}

} // namespace improc::calib
