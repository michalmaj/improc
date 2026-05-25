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
    return {};
}

LineSet DetectLines::operator()(Image<Gray> img) const {
    return {};
}

QRResult DetectQR::operator()(Image<BGR> img) const {
    return {};
}

BarcodeResult DetectBarcode::operator()(Image<BGR> img) const {
    return {};
}

} // namespace improc::core
