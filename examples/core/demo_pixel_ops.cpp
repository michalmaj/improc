// examples/core/demo_pixel_ops.cpp
//
// Demo: Add, Subtract, Multiply, AbsDiff, SplitChannels, MergeChannels

#include <format>
#include <iostream>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

int main() {
    cv::Mat m(50, 50, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(m);
    cv::Mat addend(50, 50, CV_8UC3, cv::Scalar(10, 20, 30));

    // ── Arithmetic ────────────────────────────────────────────────────────────
    Image<BGR> added  = img | Add(addend);
    Image<BGR> subbed = img | Subtract(addend);
    Image<BGR> mulled = img | Multiply(cv::Mat(50, 50, CV_8UC3, cv::Scalar(2, 2, 2)));
    Image<BGR> diff   = img | AbsDiff(addend);

    std::cout << std::format("Add     B channel: {}\n", (int)added.mat().at<cv::Vec3b>(0,0)[0]);
    std::cout << std::format("Subtract B channel: {}\n", (int)subbed.mat().at<cv::Vec3b>(0,0)[0]);
    std::cout << std::format("AbsDiff B channel: {}\n", (int)diff.mat().at<cv::Vec3b>(0,0)[0]);

    // ── SplitChannels / MergeChannels ─────────────────────────────────────────
    auto [b, g, r] = SplitChannels{}(img);
    std::cout << std::format("B channel mean: {:.1f}\n", cv::mean(b.mat())[0]);
    std::cout << std::format("G channel mean: {:.1f}\n", cv::mean(g.mat())[0]);
    std::cout << std::format("R channel mean: {:.1f}\n", cv::mean(r.mat())[0]);

    Image<BGR> merged = MergeChannels{}(b, g, r);
    cv::Mat roundtrip_diff;
    cv::absdiff(img.mat(), merged.mat(), roundtrip_diff);
    cv::Mat roundtrip_gray;
    cv::cvtColor(roundtrip_diff, roundtrip_gray, cv::COLOR_BGR2GRAY);
    std::cout << std::format("Split/Merge round-trip max error: {:.1f}\n",
                             cv::norm(roundtrip_gray, cv::NORM_INF));

    std::cout << "demo_pixel_ops: OK\n";
    return 0;
}
