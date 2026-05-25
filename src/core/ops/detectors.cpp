// src/core/ops/detectors.cpp
#include "improc/core/ops/detectors.hpp"
#include <opencv2/imgproc.hpp>

namespace improc::core {

KeypointSet DetectFAST::operator()(Image<Gray> img) const {
    KeypointSet out;
    cv::FAST(img.mat(), out.keypoints, threshold_, nms_);
    return out;
}

KeypointSet DetectBlob::operator()(Image<Gray> img) const {
    KeypointSet out;
    cv::SimpleBlobDetector::create(params_)->detect(img.mat(), out.keypoints);
    return out;
}

MSERResult DetectMSER::operator()(Image<Gray> img) const {
    MSERResult out;
    cv::MSER::create(delta_, min_area_, max_area_)
        ->detectRegions(img.mat(), out.regions, out.bboxes);
    return out;
}

LineSet DetectLines::operator()(Image<Gray> img) const {
    LineSet out;
    cv::Mat lines_mat;
    cv::createLineSegmentDetector(cv::LSD_REFINE_STD, scale_, sigma_scale_)
        ->detect(img.mat(), lines_mat);
    if (lines_mat.empty())
        return out;
    out.lines.reserve(lines_mat.rows);
    for (int i = 0; i < lines_mat.rows; ++i)
        out.lines.push_back(lines_mat.at<cv::Vec4f>(i, 0));
    return out;
}

QRResult DetectQR::operator()(Image<BGR> img) const {
    QRResult out;
    cv::QRCodeDetectorAruco detector;
    cv::Mat points_mat;
    bool found = detector.detectAndDecodeMulti(img.mat(), out.decoded, points_mat);
    if (found && !points_mat.empty()) {
        // points_mat is N×4×2 (CV_32FC2); split into per-code mats
        out.points.reserve(static_cast<std::size_t>(points_mat.rows));
        for (int i = 0; i < points_mat.rows; ++i)
            out.points.push_back(points_mat.row(i));
    }
    return out;
}

BarcodeResult DetectBarcode::operator()(Image<BGR> img) const {
    BarcodeResult out;
    cv::barcode::BarcodeDetector detector;
    std::vector<cv::Point2f> raw_pts;  // 4 corners per barcode, consecutive
    bool found = detector.detectAndDecodeWithType(
        img.mat(), out.decoded, out.types, raw_pts);
    if (!found || out.decoded.empty())
        return out;
    // raw_pts layout: [barcode0_corner0..corner3, barcode1_corner0..corner3, ...]
    for (std::size_t i = 0; i < out.decoded.size(); ++i) {
        std::vector<cv::Point2f> corners(raw_pts.begin() + i * 4,
                                         raw_pts.begin() + i * 4 + 4);
        out.bboxes.push_back(cv::minAreaRect(corners));
    }
    return out;
}

} // namespace improc::core
