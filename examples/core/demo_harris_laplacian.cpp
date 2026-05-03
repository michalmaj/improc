// examples/core/demo_harris_laplacian.cpp
//
// Demo: LaplacianEdge and HarrisCorner
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
    // ── Source image: shapes on a dark background ─────────────────────────────
    cv::Mat raw(300, 400, CV_8UC3, cv::Scalar(30, 30, 30));
    cv::rectangle(raw, {50,  50},  {150, 150}, {200, 200, 200}, -1);
    cv::circle(raw,   {280, 150},  60,          {180, 180, 180}, -1);
    cv::line(raw,     {0,   250},  {400, 250},  {220, 220, 220},  3);
    cv::ellipse(raw,  {200, 240},  {80, 40}, 30, 0, 360, {160, 160, 160}, -1);

    Image<BGR>  bgr(raw);
    Image<Gray> gray = bgr | ToGray{};

    show("0a — Source (BGR)",  bgr.mat());
    show("0b — Source (Gray)", gray.mat());

    // ── 1. LaplacianEdge ──────────────────────────────────────────────────────
    std::cout << "\n--- LaplacianEdge ---\n";

    // Default (ksize=1, scale=1.0, delta=0.0)
    Image<Gray> lap_default = gray | LaplacianEdge{};
    show("1a — Laplacian default (ksize=1)", lap_default.mat());

    // Larger kernel — wider, smoother response
    Image<Gray> lap_k3 = gray | LaplacianEdge{}.ksize(3);
    show("1b — Laplacian ksize=3", lap_k3.mat());

    // Scale amplifies response; delta shifts it (useful to see negative responses)
    Image<Gray> lap_scaled = gray | LaplacianEdge{}.ksize(1).scale(2.0).delta(64.0);
    show("1c — Laplacian scale=2 delta=64 (shows negative responses as mid-grey)", lap_scaled.mat());

    // BGR input — auto-converted to Gray internally
    Image<Gray> lap_bgr = bgr | LaplacianEdge{}.ksize(1);
    show("1d — Laplacian from BGR input (auto-converted)", lap_bgr.mat());

    // Pipeline: blur first to reduce noise sensitivity
    Image<Gray> lap_clean = gray
        | GaussianBlur{}.kernel_size(3).sigma(1.0)
        | LaplacianEdge{}.ksize(1);
    show("1e — Blur → Laplacian (cleaner edges)", lap_clean.mat());

    // ── 2. HarrisCorner ───────────────────────────────────────────────────────
    std::cout << "\n--- HarrisCorner ---\n";

    // Default (block_size=2, ksize=3, k=0.04)
    Image<Gray> harris_default = gray | HarrisCorner{};
    show("2a — Harris default (block=2, ksize=3, k=0.04)", harris_default.mat());

    // Larger block_size — averages over more pixels; fewer but more stable corners
    Image<Gray> harris_b4 = gray | HarrisCorner{}.block_size(4);
    show("2b — Harris block_size=4 (larger averaging window)", harris_b4.mat());

    // Larger ksize — broader Sobel; less sensitive to fine texture
    Image<Gray> harris_k5 = gray | HarrisCorner{}.ksize(5);
    show("2c — Harris ksize=5 Sobel kernel", harris_k5.mat());

    // Sensitivity: lower k → more detections (also more false positives)
    Image<Gray> harris_low_k  = gray | HarrisCorner{}.k(0.01);
    Image<Gray> harris_high_k = gray | HarrisCorner{}.k(0.1);
    show("2d — Harris k=0.01 (sensitive)", harris_low_k.mat());
    show("2e — Harris k=0.10 (conservative)", harris_high_k.mat());

    // BGR input — auto-converted to Gray internally
    Image<Gray> harris_bgr = bgr | HarrisCorner{};
    show("2f — Harris from BGR input (auto-converted)", harris_bgr.mat());

    // ── 3. Laplacian vs Harris comparison ────────────────────────────────────
    std::cout << "\n--- Laplacian vs Harris ---\n";

    Image<Gray> prepped = gray | GaussianBlur{}.kernel_size(3).sigma(1.0);
    Image<Gray> lap_comp    = prepped | LaplacianEdge{}.ksize(1);
    Image<Gray> harris_comp = prepped | HarrisCorner{};

    show("3a — Blurred input", prepped.mat());
    show("3b — Laplacian after blur (all edges, 2nd derivative)", lap_comp.mat());
    show("3c — Harris after blur (corner responses only)", harris_comp.mat());

    // ── 4. Pipeline integration ───────────────────────────────────────────────
    std::cout << "\n--- Pipeline ---\n";

    // Threshold the Harris response to get a binary corner mask
    Image<Gray> corner_mask = gray
        | HarrisCorner{}.block_size(2).ksize(3).k(0.04)
        | Threshold{}.value(100).mode(ThresholdMode::Binary);
    show("4a — Harris → Binary threshold (strong corners only)", corner_mask.mat());

    // Laplacian → Canny: detect edges in the second-derivative map
    Image<Gray> lap_canny = gray
        | LaplacianEdge{}.ksize(1)
        | CannyEdge{}.threshold1(30).threshold2(90);
    show("4b — Laplacian → Canny (thin edges from 2nd derivative)", lap_canny.mat());

    std::cout << "Done.\n";
    return 0;
}
