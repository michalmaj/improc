// examples/core/demo_detectors.cpp
//
// Demo: DetectFAST, DetectBlob, DetectMSER, DetectLines
//
// Generates a synthetic scene with rectangles and circles, runs all four
// detectors, draws results in named windows.
// Press any key to advance.

#include "improc/core/pipeline.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace improc::core;

static void show(const std::string& t, const cv::Mat& m) {
    cv::imshow(t, m); cv::waitKey(0);
}

int main() {
    // ── Synthetic scene ───────────────────────────────────────────────────────
    cv::Mat raw(400, 600, CV_8UC3, cv::Scalar(50, 50, 50));
    cv::rectangle(raw, {40,  40},  {200, 200}, {200, 200, 200}, -1);
    cv::rectangle(raw, {240, 60},  {400, 180}, {160, 160, 160}, 3);
    cv::circle(raw,    {500, 120}, 70,          {220, 220, 220}, -1);
    cv::circle(raw,    {100, 300}, 50,          {180, 180, 180}, 4);
    cv::line(raw,      {0,   350}, {600, 350},  {200, 200, 200}, 2);
    cv::line(raw,      {300,   0}, {450, 400},  {170, 170, 170}, 2);

    Image<BGR>  bgr(raw);
    Image<Gray> gray = bgr | ToGray{};

    show("0 — Source", raw);

    // ── DetectFAST ────────────────────────────────────────────────────────────
    std::cout << "\n--- DetectFAST ---\n";
    KeypointSet fast_kps = gray | DetectFAST{}
        .threshold(10)              // contrast threshold (default: 10)
        .non_max_suppression(true); // suppress adjacent responses (default: true)
    std::cout << "FAST corners: " << fast_kps.size() << "\n";

    cv::Mat fast_vis = raw.clone();
    cv::drawKeypoints(fast_vis, fast_kps.keypoints, fast_vis,
                      {0, 255, 0}, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    show("1 — FAST corners", fast_vis);

    // ── DetectBlob ────────────────────────────────────────────────────────────
    std::cout << "\n--- DetectBlob ---\n";
    cv::SimpleBlobDetector::Params params;
    params.minArea        = 200.f;
    params.maxArea        = 30000.f;
    params.minCircularity = 0.5f;
    params.filterByConvexity = false;
    params.filterByInertia   = false;

    KeypointSet blobs = gray | DetectBlob{}.params(params);
    std::cout << "Blobs: " << blobs.size() << "\n";

    cv::Mat blob_vis = raw.clone();
    cv::drawKeypoints(blob_vis, blobs.keypoints, blob_vis,
                      {0, 0, 255}, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    show("2 — Blobs (red circles, size = diameter)", blob_vis);

    // ── DetectMSER ────────────────────────────────────────────────────────────
    std::cout << "\n--- DetectMSER ---\n";
    MSERResult mser = gray | DetectMSER{}
        .delta(5)         // stability threshold (default: 5)
        .min_area(200)    // minimum region area in px (default: 60)
        .max_area(50000); // maximum region area in px (default: 14400)
    std::cout << "MSER regions: " << mser.size() << "\n";

    cv::Mat mser_vis = raw.clone();
    for (const auto& bbox : mser.bboxes)
        cv::rectangle(mser_vis, bbox, {255, 0, 0}, 2);
    show("3 — MSER bounding boxes (blue)", mser_vis);

    // ── DetectLines ───────────────────────────────────────────────────────────
    std::cout << "\n--- DetectLines ---\n";
    LineSet ls = gray | DetectLines{}
        .scale(0.8)        // image scale for detection (default: 0.8)
        .sigma_scale(0.6); // Gaussian blur scale (default: 0.6)
    std::cout << "Lines: " << ls.size() << "\n";

    cv::Mat line_vis = raw.clone();
    for (const auto& l : ls.lines)
        cv::line(line_vis,
                 {static_cast<int>(l[0]), static_cast<int>(l[1])},
                 {static_cast<int>(l[2]), static_cast<int>(l[3])},
                 {0, 165, 255}, 2);
    show("4 — LSD line segments (orange)", line_vis);

    std::cout << "Done.\n";
    return 0;
}
