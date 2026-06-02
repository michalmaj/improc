// include/improc/core/ops/detectors.hpp
#pragma once
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include "improc/core/image.hpp"
#include "improc/core/ops/feature_detection.hpp"  // KeypointSet
#include "improc/core/ops/detector_types.hpp"

namespace improc::core {

/// @brief Pipeline op: detects FAST corners in `Image<Gray>`. Returns `KeypointSet`.
/// Fluent setters: `threshold(t)` (default 10), `non_max_suppression(b)` (default true).
/// @code
/// KeypointSet kps = gray | DetectFAST{}.threshold(20);
/// @endcode
struct DetectFAST {
    DetectFAST& threshold(int t)             { threshold_ = t;   return *this; }
    DetectFAST& non_max_suppression(bool nms) { nms_       = nms; return *this; }
    [[nodiscard]] KeypointSet operator()(Image<Gray> img) const;
private:
    int  threshold_ = 10;
    bool nms_       = true;
};

/// @brief Pipeline op: detects blobs in `Image<Gray>` using `cv::SimpleBlobDetector`.
/// Returns `KeypointSet`. Default params detect dark blobs (area 25–5000, circularity ≥ 0.8).
/// @code
/// KeypointSet kps = gray | DetectBlob{};
/// @endcode
struct DetectBlob {
    DetectBlob& params(cv::SimpleBlobDetector::Params p) { params_ = p; return *this; }
    [[nodiscard]] KeypointSet operator()(Image<Gray> img) const;
private:
    cv::SimpleBlobDetector::Params params_;
};

/// @brief Pipeline op: detects MSER stable regions in `Image<Gray>`. Returns `MSERResult`.
/// Fluent setters: `delta(d)` (default 5), `min_area(a)` (default 60), `max_area(a)` (default 14400).
/// @code
/// MSERResult r = gray | DetectMSER{}.min_area(100);
/// @endcode
struct DetectMSER {
    DetectMSER& delta(int d)    { delta_    = d; return *this; }
    DetectMSER& min_area(int a) { min_area_ = a; return *this; }
    DetectMSER& max_area(int a) { max_area_ = a; return *this; }
    [[nodiscard]] MSERResult  operator()(Image<Gray> img) const;
private:
    int delta_    = 5;
    int min_area_ = 60;
    int max_area_ = 14400;
};

/// @brief Pipeline op: detects line segments in `Image<Gray>` using LSD. Returns `LineSet`.
/// Each entry in `lines` is `(x1, y1, x2, y2)`. Fluent setters: `scale(s)` (default 0.8), `sigma_scale(s)` (default 0.6).
/// @code
/// LineSet ls = gray | DetectLines{};
/// @endcode
struct DetectLines {
    DetectLines& scale(double s)       { scale_       = s; return *this; }
    DetectLines& sigma_scale(double s) { sigma_scale_ = s; return *this; }
    [[nodiscard]] LineSet      operator()(Image<Gray> img) const;
private:
    double scale_       = 0.8;
    double sigma_scale_ = 0.6;
};

/// @brief Pipeline op: detects and decodes QR codes in `Image<BGR>`. Returns `QRResult`.
/// @code
/// QRResult qr = bgr | DetectQR{};
/// @endcode
struct DetectQR {
    [[nodiscard]] QRResult operator()(Image<BGR> img) const;
};

/// @brief Pipeline op: detects and decodes barcodes in `Image<BGR>` (no model file needed). Returns `BarcodeResult`.
/// @code
/// BarcodeResult bc = bgr | DetectBarcode{};
/// @endcode
struct DetectBarcode {
    [[nodiscard]] BarcodeResult operator()(Image<BGR> img) const;
};

} // namespace improc::core
