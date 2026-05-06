// examples/core/demo_feature_detection.cpp
//
// Demo: DetectORB, DetectSIFT, DetectAKAZE, KeypointSet
//
// Usage: run from the build directory; press any key to advance.

#include "improc/core/pipeline.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <format>
#include <iostream>

using namespace improc::core;

static void show(const std::string& title, const cv::Mat& mat) {
    cv::imshow(title, mat);
    cv::waitKey(0);
}

static cv::Mat draw_keypoints(const cv::Mat& src, const KeypointSet& ks,
                              cv::Scalar color = {0, 255, 0}) {
    cv::Mat out;
    cv::drawKeypoints(src, ks.keypoints, out, color,
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return out;
}

int main() {
    // ── Source: synthetic scene with rich corners ─────────────────────────────
    cv::Mat raw(300, 400, CV_8UC3, cv::Scalar(80, 80, 80));
    cv::rectangle(raw, {20,  20},  {150, 150}, cv::Scalar(220, 220, 220), -1);
    cv::rectangle(raw, {180, 20},  {360, 100}, cv::Scalar(30,  30,  30),  -1);
    cv::circle(raw,    {100, 230}, 60,          cv::Scalar(200, 160, 80),  -1);
    cv::ellipse(raw,   {300, 220}, {70, 40}, 30, 0, 360, cv::Scalar(50, 100, 200), -1);
    Image<BGR>  bgr(raw);
    Image<Gray> gray = bgr | ToGray{};

    show("0 — Source BGR", bgr.mat());

    // ── ORB ───────────────────────────────────────────────────────────────────
    KeypointSet orb_ks = gray | DetectORB{};
    std::cout << std::format("ORB (default 500): {} keypoints\n", orb_ks.size());
    show("1a — ORB (default)", draw_keypoints(raw, orb_ks, {0, 255, 0}));

    KeypointSet orb_50 = gray | DetectORB{}.max_features(50);
    std::cout << std::format("ORB (max 50): {} keypoints\n", orb_50.size());
    show("1b — ORB (max_features=50)", draw_keypoints(raw, orb_50, {0, 200, 0}));

    // ── SIFT ──────────────────────────────────────────────────────────────────
    KeypointSet sift_ks = gray | DetectSIFT{};
    std::cout << std::format("SIFT (default): {} keypoints\n", sift_ks.size());
    show("2a — SIFT (default)", draw_keypoints(raw, sift_ks, {0, 0, 255}));

    KeypointSet sift_50 = gray | DetectSIFT{}.max_features(50);
    std::cout << std::format("SIFT (max 50): {} keypoints\n", sift_50.size());
    show("2b — SIFT (max_features=50)", draw_keypoints(raw, sift_50, {0, 0, 200}));

    // ── AKAZE ─────────────────────────────────────────────────────────────────
    KeypointSet akaze_ks = gray | DetectAKAZE{};
    std::cout << std::format("AKAZE (threshold=0.001): {} keypoints\n", akaze_ks.size());
    show("3a — AKAZE (default threshold)", draw_keypoints(raw, akaze_ks, {255, 0, 0}));

    KeypointSet akaze_strict = gray | DetectAKAZE{}.threshold(0.01f);
    std::cout << std::format("AKAZE (threshold=0.01):  {} keypoints\n", akaze_strict.size());
    show("3b — AKAZE (strict threshold)", draw_keypoints(raw, akaze_strict, {200, 0, 0}));

    // ── Pipeline comparison ───────────────────────────────────────────────────
    auto ks_o = bgr | ToGray{} | DetectORB{}.max_features(100);
    auto ks_s = bgr | ToGray{} | DetectSIFT{}.max_features(100);
    auto ks_a = bgr | ToGray{} | DetectAKAZE{};
    std::cout << std::format("\nPipeline (max 100 each):\n  ORB={} SIFT={} AKAZE={}\n",
        ks_o.size(), ks_s.size(), ks_a.size());

    std::cout << "Done.\n";
    return 0;
}
