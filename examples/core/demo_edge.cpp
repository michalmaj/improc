// examples/core/demo_edge.cpp
//
// Demo: SobelEdge and CannyEdge
//
// Usage: run from the build directory; press any key to advance.

#include "improc/core/pipeline.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace improc::core;

static void show(const std::string& title, const cv::Mat& mat) {
    cv::imshow(title, mat);
    cv::waitKey(0);
}

int main() {
    // ── Source images ──────────────────────────────────────────────────────────
    // Synthetic: shapes on a dark background
    cv::Mat raw(300, 400, CV_8UC3, cv::Scalar(30, 30, 30));
    cv::rectangle(raw, {50, 50},   {150, 150}, {200, 200, 200}, -1);
    cv::circle(raw,   {280, 150},  60,          {180, 180, 180}, -1);
    cv::line(raw,     {0,  250},   {400, 250},  {220, 220, 220},  3);
    cv::ellipse(raw,  {200, 240},  {80, 40}, 30, 0, 360, {160, 160, 160}, -1);

    Image<BGR>  bgr(raw);
    Image<Gray> gray = bgr | ToGray{};

    show("0a — Source (BGR)", bgr.mat());
    show("0b — Source (Gray)", gray.mat());

    // ── 1. SobelEdge ──────────────────────────────────────────────────────────
    std::cout << "\n--- SobelEdge ---\n";

    Image<Gray> sobel3 = gray | SobelEdge{}.ksize(3);
    Image<Gray> sobel5 = gray | SobelEdge{}.ksize(5);
    Image<Gray> sobel7 = gray | SobelEdge{}.ksize(7);
    show("1a — Sobel ksize=3 (fine detail)", sobel3.mat());
    show("1b — Sobel ksize=5 (smoother)", sobel5.mat());
    show("1c — Sobel ksize=7 (broadest edges)", sobel7.mat());

    // BGR input — auto-converted to Gray internally
    Image<Gray> sobel_bgr = bgr | SobelEdge{};
    show("1d — Sobel from BGR input (auto-converted)", sobel_bgr.mat());

    // ── 2. CannyEdge ──────────────────────────────────────────────────────────
    std::cout << "\n--- CannyEdge ---\n";

    Image<Gray> canny_default = gray | CannyEdge{};
    Image<Gray> canny_low     = gray | CannyEdge{}.threshold1(20).threshold2(60);
    Image<Gray> canny_high    = gray | CannyEdge{}.threshold1(150).threshold2(250);
    Image<Gray> canny_ap5     = gray | CannyEdge{}.threshold1(50).threshold2(150).aperture_size(5);
    show("2a — Canny (t1=100, t2=200) default", canny_default.mat());
    show("2b — Canny (t1=20, t2=60) low thresholds — more edges", canny_low.mat());
    show("2c — Canny (t1=150, t2=250) high thresholds — fewer edges", canny_high.mat());
    show("2d — Canny (aperture=5) — slightly smoother edges", canny_ap5.mat());

    // BGR input
    Image<Gray> canny_bgr = bgr | CannyEdge{}.threshold1(50).threshold2(150);
    show("2e — Canny from BGR input (auto-converted)", canny_bgr.mat());

    // ── 3. Sobel vs Canny comparison ──────────────────────────────────────────
    std::cout << "\n--- Sobel vs Canny comparison ---\n";

    // Blur first to reduce noise sensitivity
    Image<Gray> prepped = gray | GaussianBlur{}.kernel_size(3).sigma(1.0);
    Image<Gray> sobel_clean = prepped | SobelEdge{}.ksize(3);
    Image<Gray> canny_clean = prepped | CannyEdge{}.threshold1(40).threshold2(120);
    show("3a — Blurred input", prepped.mat());
    show("3b — Sobel after blur (gradient magnitude)", sobel_clean.mat());
    show("3c — Canny after blur (thin binary edges)", canny_clean.mat());

    // ── 4. Pipeline integration ───────────────────────────────────────────────
    std::cout << "\n--- Pipeline: BGR → CLAHE → Sobel → Threshold ---\n";

    Image<Gray> pipeline_result = bgr
        | CLAHE{}.clip_limit(2.0)
        | SobelEdge{}.ksize(3)
        | Threshold{}.value(30).mode(ThresholdMode::Binary);

    show("4  — BGR → CLAHE → Sobel → Binary threshold", pipeline_result.mat());

    std::cout << "Done.\n";
    return 0;
}
