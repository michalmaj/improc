// include/improc/core/ops/detector_types.hpp
#pragma once
#include <array>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace improc::core {

// ── MSER ─────────────────────────────────────────────────────────────────────
/**
 * @brief Result of MSER (Maximally Stable Extremal Regions) detection.
 */
struct MSERResult {
    std::vector<std::vector<cv::Point>> regions; ///< Pixel sets, one per detected region.
    std::vector<cv::Rect>               bboxes;  ///< Axis-aligned bounding box per region.
    std::size_t size()  const { return regions.size(); }
    bool        empty() const { return regions.empty(); }
};

// ── LSD lines ────────────────────────────────────────────────────────────────
/**
 * @brief Result of LSD (Line Segment Detector) line detection.
 */
struct LineSet {
    std::vector<cv::Vec4f> lines; ///< Each entry: (x1, y1, x2, y2) in pixel coordinates.
    std::size_t size()  const { return lines.size(); }
    bool        empty() const { return lines.empty(); }
};

// ── QR codes ─────────────────────────────────────────────────────────────────
/**
 * @brief Result of QR code detection and decoding.
 */
struct QRResult {
    std::vector<std::string> decoded; ///< Decoded content strings, one per detected code.
    std::vector<cv::Mat>     points;  ///< 4-corner matrices (CV_32FC2, 4×1) per code.
    std::size_t size()  const { return decoded.size(); }
    bool        empty() const { return decoded.empty(); }
};

// ── Barcodes ─────────────────────────────────────────────────────────────────
/**
 * @brief Result of barcode detection and decoding.
 */
struct BarcodeResult {
    std::vector<std::string>     decoded; ///< Decoded content strings.
    std::vector<std::string>     types;   ///< Format name per decoded entry (e.g. "EAN_13").
    std::vector<cv::RotatedRect> bboxes;  ///< Oriented bounding box per entry.
    std::size_t size()  const { return decoded.size(); }
    bool        empty() const { return decoded.empty(); }
};

// ── Face detection ───────────────────────────────────────────────────────────
/**
 * @brief Single face detection with landmark points.
 */
struct FaceDetection {
    cv::Rect2f                 bbox;       ///< Face bounding box in float pixel coordinates.
    float                      confidence; ///< Detection confidence score in [0, 1].
    std::array<cv::Point2f, 5> landmarks;  ///< Facial landmarks: right-eye, left-eye, nose, right-mouth, left-mouth.
};

} // namespace improc::core
