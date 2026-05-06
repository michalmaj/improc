// examples/core/demo_describe.cpp
//
// Demo: DescribeORB, DescribeSIFT, DescribeAKAZE, DescriptorSet
//
// Usage: run from the build directory; no display window needed.

#include "improc/core/pipeline.hpp"
#include <opencv2/imgproc.hpp>
#include <format>
#include <iostream>

using namespace improc::core;

int main() {
    cv::Mat raw(300, 400, CV_8UC3, cv::Scalar(80, 80, 80));
    cv::rectangle(raw, {20,  20},  {150, 150}, cv::Scalar(220, 220, 220), -1);
    cv::rectangle(raw, {180, 20},  {360, 100}, cv::Scalar(30,  30,  30),  -1);
    cv::circle(raw,    {100, 230}, 60,          cv::Scalar(200, 160, 80),  -1);
    cv::ellipse(raw,   {300, 220}, {70, 40}, 30, 0, 360, cv::Scalar(50, 100, 200), -1);

    Image<BGR>  bgr(raw);
    Image<Gray> gray = bgr | ToGray{};

    // ── ORB: detect → describe ────────────────────────────────────────────────
    KeypointSet kps_orb    = gray | DetectORB{}.max_features(200);
    DescriptorSet desc_orb = gray | DescribeORB{kps_orb};
    std::cout << std::format("ORB:   {} keypoints → {} descriptors (CV_8U={}): {}\n",
        kps_orb.size(), desc_orb.size(),
        CV_8U, desc_orb.descriptors.type() == CV_8U ? "OK" : "FAIL");

    DescriptorSet desc_orb_bgr = bgr | DescribeORB{kps_orb};
    std::cout << std::format("ORB (BGR input): {} descriptors\n", desc_orb_bgr.size());

    // ── SIFT: detect → describe ───────────────────────────────────────────────
    KeypointSet kps_sift    = gray | DetectSIFT{};
    DescriptorSet desc_sift = gray | DescribeSIFT{kps_sift};
    std::cout << std::format("SIFT:  {} keypoints → {} descriptors (CV_32F={}): {}\n",
        kps_sift.size(), desc_sift.size(),
        CV_32F, desc_sift.descriptors.type() == CV_32F ? "OK" : "FAIL");

    // ── AKAZE: detect → describe ──────────────────────────────────────────────
    KeypointSet kps_akaze    = gray | DetectAKAZE{};
    DescriptorSet desc_akaze = gray | DescribeAKAZE{kps_akaze};
    std::cout << std::format("AKAZE: {} keypoints → {} descriptors (CV_8U={}): {}\n",
        kps_akaze.size(), desc_akaze.size(),
        CV_8U, desc_akaze.descriptors.type() == CV_8U ? "OK" : "FAIL");

    // ── Pipeline form ─────────────────────────────────────────────────────────
    auto kps2  = bgr | ToGray{} | DetectORB{}.max_features(100);
    auto desc2 = bgr | ToGray{} | DescribeORB{kps2};
    auto desc3 = bgr | DescribeORB{kps2};
    std::cout << std::format("\nPipeline: {} descriptors (Gray), {} descriptors (BGR)\n",
        desc2.size(), desc3.size());

    // ── Empty keypoints edge case ─────────────────────────────────────────────
    DescriptorSet empty_ds = gray | DescribeORB{KeypointSet{}};
    std::cout << std::format("Empty input: size={} empty={}\n",
        empty_ds.size(), empty_ds.empty());

    std::cout << "Done.\n";
    return 0;
}
