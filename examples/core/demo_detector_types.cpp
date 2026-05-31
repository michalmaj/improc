// examples/core/demo_detector_types.cpp
//
// Demo: MSERResult, LineSet, QRResult, BarcodeResult, FaceDetection

#include <format>
#include <iostream>
#include "improc/core/ops/detector_types.hpp"

using namespace improc::core;

int main() {
    // ── MSERResult ────────────────────────────────────────────────────────────
    MSERResult mser;
    mser.regions.push_back({{0,0},{1,1},{2,2}});
    mser.bboxes.push_back(cv::Rect{0, 0, 3, 3});
    std::cout << std::format("MSERResult  size={} empty={}\n",
                             mser.size(), mser.empty() ? "true" : "false");

    // ── LineSet ───────────────────────────────────────────────────────────────
    LineSet ls;
    ls.lines.push_back({0.f, 0.f, 100.f, 100.f});
    std::cout << std::format("LineSet     size={} first=({},{})\n",
                             ls.size(), ls.lines[0][0], ls.lines[0][1]);

    // ── QRResult ──────────────────────────────────────────────────────────────
    QRResult qr;
    qr.decoded.push_back("https://example.com");
    qr.points.push_back(cv::Mat());
    std::cout << std::format("QRResult    size={} decoded='{}'\n",
                             qr.size(), qr.decoded[0]);

    // ── BarcodeResult ─────────────────────────────────────────────────────────
    BarcodeResult bc;
    bc.decoded.push_back("9780201379624");
    bc.types.push_back("EAN_13");
    bc.bboxes.push_back(cv::RotatedRect{});
    std::cout << std::format("BarcodeResult size={} type={}\n",
                             bc.size(), bc.types[0]);

    // ── FaceDetection ─────────────────────────────────────────────────────────
    FaceDetection fd;
    fd.bbox       = cv::Rect2f{10.f, 10.f, 80.f, 100.f};
    fd.confidence = 0.98f;
    fd.landmarks[0] = cv::Point2f{30.f, 35.f};
    std::cout << std::format("FaceDetection confidence={:.2f} bbox=({},{},{},{})\n",
                             fd.confidence, fd.bbox.x, fd.bbox.y,
                             fd.bbox.width, fd.bbox.height);

    std::cout << "demo_detector_types: OK\n";
    return 0;
}
