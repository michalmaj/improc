// include/improc/core/ops/detectors.hpp
#pragma once
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include "improc/core/image.hpp"
#include "improc/core/ops/feature_detection.hpp"  // KeypointSet
#include "improc/core/ops/detector_types.hpp"

namespace improc::core {

// ── DetectFAST ────────────────────────────────────────────────────────────────
struct DetectFAST {
    DetectFAST& threshold(int t)             { threshold_ = t;   return *this; }
    DetectFAST& nonmax_suppression(bool nms) { nms_       = nms; return *this; }
    KeypointSet operator()(Image<Gray> img) const;
private:
    int  threshold_ = 10;
    bool nms_       = true;
};

// ── DetectBlob ───────────────────────────────────────────────────────────────
struct DetectBlob {
    DetectBlob& params(cv::SimpleBlobDetector::Params p) { params_ = p; return *this; }
    KeypointSet operator()(Image<Gray> img) const;
private:
    cv::SimpleBlobDetector::Params params_;
};

// ── DetectMSER ───────────────────────────────────────────────────────────────
struct DetectMSER {
    DetectMSER& delta(int d)    { delta_    = d; return *this; }
    DetectMSER& min_area(int a) { min_area_ = a; return *this; }
    DetectMSER& max_area(int a) { max_area_ = a; return *this; }
    MSERResult  operator()(Image<Gray> img) const;
private:
    int delta_    = 5;
    int min_area_ = 60;
    int max_area_ = 14400;
};

// ── DetectLines ──────────────────────────────────────────────────────────────
struct DetectLines {
    DetectLines& scale(double s)       { scale_       = s; return *this; }
    DetectLines& sigma_scale(double s) { sigma_scale_ = s; return *this; }
    LineSet      operator()(Image<Gray> img) const;
private:
    double scale_       = 0.8;
    double sigma_scale_ = 0.6;
};

// ── DetectQR ─────────────────────────────────────────────────────────────────
struct DetectQR {
    QRResult operator()(Image<BGR> img) const;
};

// ── DetectBarcode ────────────────────────────────────────────────────────────
struct DetectBarcode {
    BarcodeResult operator()(Image<BGR> img) const;
};

} // namespace improc::core
