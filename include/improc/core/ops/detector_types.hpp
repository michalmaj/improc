// include/improc/core/ops/detector_types.hpp
#pragma once
#include <array>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace improc::core {

// ── MSER ─────────────────────────────────────────────────────────────────────
struct MSERResult {
    std::vector<std::vector<cv::Point>> regions;
    std::vector<cv::Rect>               bboxes;
    std::size_t size()  const { return regions.size(); }
    bool        empty() const { return regions.empty(); }
};

// ── LSD lines ────────────────────────────────────────────────────────────────
struct LineSet {
    std::vector<cv::Vec4f> lines; ///< each entry: (x1, y1, x2, y2)
    std::size_t size()  const { return lines.size(); }
    bool        empty() const { return lines.empty(); }
};

// ── QR codes ─────────────────────────────────────────────────────────────────
struct QRResult {
    std::vector<std::string> decoded; ///< one decoded string per QR code
    std::vector<cv::Mat>     points;  ///< 4-corner mat (CV_32FC2, 4×1) per code
    std::size_t size()  const { return decoded.size(); }
    bool        empty() const { return decoded.empty(); }
};

// ── Barcodes ─────────────────────────────────────────────────────────────────
struct BarcodeResult {
    std::vector<std::string>     decoded; ///< one decoded string per barcode
    std::vector<std::string>     types;   ///< barcode format name per entry
    std::vector<cv::RotatedRect> bboxes;  ///< oriented bbox per entry
    std::size_t size()  const { return decoded.size(); }
    bool        empty() const { return decoded.empty(); }
};

// ── Face detection ───────────────────────────────────────────────────────────
struct FaceDetection {
    cv::Rect2f                 bbox;        ///< face bounding box (float coords)
    float                      confidence;  ///< detection score
    std::array<cv::Point2f, 5> landmarks;   ///< right-eye, left-eye, nose, right-mouth, left-mouth
};

} // namespace improc::core
