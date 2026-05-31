// examples/core/demo_morphological_ops.cpp
//
// Demo: FloodFill, GrabCut, Watershed, DistanceTransform, Inpaint

#include <format>
#include <iostream>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

int main() {
    // ── FloodFill ─────────────────────────────────────────────────────────────
    cv::Mat m_fill(100, 100, CV_8UC3, cv::Scalar(200, 200, 200));
    Image<BGR> fill_src(m_fill);
    Image<BGR> filled = FloodFill()
        .lo_diff({20, 20, 20})
        .up_diff({20, 20, 20})(fill_src, {50, 50}, {0, 0, 255});
    auto seed_px = filled.mat().at<cv::Vec3b>(50, 50);
    std::cout << std::format("FloodFill seed R={} (expected 255)\n", (int)seed_px[2]);

    // ── GrabCut ───────────────────────────────────────────────────────────────
    cv::Mat m_gc(200, 200, CV_8UC3, cv::Scalar(80, 80, 80));
    cv::rectangle(m_gc, {60, 60}, {140, 140}, cv::Scalar(200, 100, 50), -1);
    Image<BGR> gc_src(m_gc);
    Image<Gray> fg_mask = GrabCut().iterations(2)(gc_src, cv::Rect{55, 55, 90, 90});
    std::cout << std::format("GrabCut foreground pixels: {}\n",
                             cv::countNonZero(fg_mask.mat()));

    // ── DistanceTransform ─────────────────────────────────────────────────────
    cv::Mat binary(100, 100, CV_8UC1, cv::Scalar(255));
    cv::rectangle(binary, {10, 10}, {90, 90}, cv::Scalar(0), 2);
    Image<Gray> bin_img(binary);
    Image<Float32> dt = bin_img | DistanceTransform{};
    std::cout << std::format("DistanceTransform max value (centre): {:.1f}\n",
                             dt.mat().at<float>(50, 50));

    // ── Watershed ─────────────────────────────────────────────────────────────
    cv::Mat wm(200, 200, CV_8UC3, cv::Scalar(80, 80, 80));
    cv::rectangle(wm, {20, 20}, {80, 80}, cv::Scalar(200, 80, 80), -1);
    cv::rectangle(wm, {110, 110}, {180, 180}, cv::Scalar(80, 200, 80), -1);
    Image<BGR> ws_src(wm);
    cv::Mat markers = cv::Mat::zeros(ws_src.mat().size(), CV_32S);
    markers.at<int>(50, 50)   = 1;
    markers.at<int>(145, 145) = 2;
    Watershed{}(ws_src, markers);
    std::cout << std::format("Watershed boundary pixels: {}\n",
                             cv::countNonZero(markers == -1));

    // ── Inpaint ───────────────────────────────────────────────────────────────
    cv::Mat ip_src(100, 100, CV_8UC3, cv::Scalar(180, 180, 180));
    cv::line(ip_src, {0, 50}, {100, 50}, cv::Scalar(0, 0, 0), 3);
    Image<BGR> ip_img(ip_src);
    cv::Mat mask_mat(100, 100, CV_8UC1, cv::Scalar(0));
    cv::line(mask_mat, {0, 50}, {100, 50}, cv::Scalar(255), 3);
    Image<Gray> ip_mask(mask_mat);
    Image<BGR> restored = Inpaint()
        .radius(5.0)
        .method(InpaintMethod::TELEA)(ip_img, ip_mask);
    std::cout << std::format("Inpaint output size: {}x{}\n",
                             restored.cols(), restored.rows());

    std::cout << "demo_morphological_ops: OK\n";
    return 0;
}
