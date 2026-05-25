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
    return {};
}

BarcodeResult DetectBarcode::operator()(Image<BGR> img) const {
    return {};
}

} // namespace improc::core
