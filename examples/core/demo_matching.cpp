// examples/core/demo_matching.cpp
//
// Demo: MatchBF, MatchFlann, MatchSet
//
// Usage: run from the build directory; no display window needed.

#include "improc/core/pipeline.hpp"
#include <opencv2/imgproc.hpp>
#include <format>
#include <iostream>

using namespace improc::core;
using improc::ParameterError;

int main() {
    cv::Mat raw(300, 400, CV_8UC3, cv::Scalar(80, 80, 80));
    cv::rectangle(raw, {20,  20},  {150, 150}, cv::Scalar(220, 220, 220), -1);
    cv::rectangle(raw, {180, 20},  {360, 100}, cv::Scalar(30,  30,  30),  -1);
    cv::circle(raw,    {100, 230}, 60,          cv::Scalar(200, 160, 80),  -1);
    cv::ellipse(raw,   {300, 220}, {70, 40}, 30, 0, 360, cv::Scalar(50, 100, 200), -1);

    Image<BGR>  bgr(raw);
    Image<Gray> gray = bgr | ToGray{};

    // ── ORB: detect → describe → BF match ────────────────────────────────────
    KeypointSet   kps_orb  = gray | DetectORB{}.max_features(200);
    DescriptorSet desc_orb = gray | DescribeORB{kps_orb};

    MatchSet ms_orb = MatchBF{desc_orb, desc_orb}();
    std::cout << std::format("ORB BF (self-match): {} matches\n", ms_orb.size());

    MatchSet ms_cross = MatchBF{desc_orb, desc_orb}.cross_check(true)();
    std::cout << std::format("ORB BF + cross_check: {} matches\n", ms_cross.size());

    MatchSet ms_dist = MatchBF{desc_orb, desc_orb}.max_distance(30.0f)();
    std::cout << std::format("ORB BF + max_distance=30: {} matches\n", ms_dist.size());

    // ── SIFT: detect → describe → BF + FLANN match ───────────────────────────
    KeypointSet   kps_sift  = gray | DetectSIFT{};
    DescriptorSet desc_sift = gray | DescribeSIFT{kps_sift};

    MatchSet ms_sift_bf = MatchBF{desc_sift, desc_sift}();
    std::cout << std::format("\nSIFT BF (self-match): {} matches\n", ms_sift_bf.size());

    MatchSet ms_flann = MatchFlann{desc_sift, desc_sift}();
    std::cout << std::format("SIFT FLANN (ratio=0.7): {} matches\n", ms_flann.size());

    MatchSet ms_tight = MatchFlann{desc_sift, desc_sift}.ratio_threshold(0.5f)();
    MatchSet ms_loose = MatchFlann{desc_sift, desc_sift}.ratio_threshold(0.9f)();
    std::cout << std::format("SIFT FLANN ratio=0.5: {} | ratio=0.9: {}\n",
        ms_tight.size(), ms_loose.size());

    // ── ParameterError demos ──────────────────────────────────────────────────
    try {
        MatchBF{desc_orb, desc_orb}.max_distance(-1.0f);
    } catch (const ParameterError& e) {
        std::cout << std::format("\nMatchBF max_distance(-1): caught ParameterError\n");
    }

    try {
        MatchFlann{desc_orb, desc_orb}();  // binary descriptors
    } catch (const ParameterError& e) {
        std::cout << std::format("MatchFlann on ORB: caught ParameterError\n");
    }

    // ── Distance range of BF matches ─────────────────────────────────────────
    if (!ms_orb.empty()) {
        float min_d = ms_orb.matches[0].distance;
        float max_d = ms_orb.matches[0].distance;
        for (auto& m : ms_orb.matches) {
            min_d = std::min(min_d, m.distance);
            max_d = std::max(max_d, m.distance);
        }
        std::cout << std::format("\nORB match distances: min={:.1f} max={:.1f}\n", min_d, max_d);
    }

    std::cout << "Done.\n";
    return 0;
}
