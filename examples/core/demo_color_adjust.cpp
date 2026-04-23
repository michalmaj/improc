// examples/core/demo_color_adjust.cpp
//
// Demo: ToHSV/ToBGR, Brightness, Contrast, WeightedBlend, AlphaBlend

#include <iostream>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

int main() {
    // --- HSV color space ---
    cv::Mat raw(4, 4, CV_8UC3, cv::Scalar(0, 0, 255)); // pure red BGR
    Image<BGR> bgr(raw);

    Image<HSV> hsv = bgr | ToHSV{};
    Image<BGR> back = hsv | ToBGR{};
    cv::Mat diff;
    cv::absdiff(bgr.mat(), back.mat(), diff);
    cv::Mat diff_gray;
    cv::cvtColor(diff, diff_gray, cv::COLOR_BGR2GRAY);
    std::cout << "HSV round-trip: original == restored? "
              << (cv::countNonZero(diff_gray) == 0 ? "yes" : "no")
              << "\n";

    // --- Brightness ---
    Image<BGR> bright = bgr | Brightness{}.delta(50.0);
    Image<BGR> dark   = bgr | Brightness{}.delta(-50.0);
    std::cout << "Brightness +50 mean: " << cv::mean(bright.mat())[2] << "\n"; // R channel
    std::cout << "Brightness -50 mean: " << cv::mean(dark.mat())[2]   << "\n";

    // --- Contrast ---
    cv::Mat mid(4, 4, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> mid_img(mid);
    Image<BGR> more = mid_img | Contrast{}.factor(1.5);
    Image<BGR> less = mid_img | Contrast{}.factor(0.5);
    std::cout << "Contrast x1.5 mean: " << cv::mean(more.mat())[0] << "\n";
    std::cout << "Contrast x0.5 mean: " << cv::mean(less.mat())[0] << "\n";

    // --- WeightedBlend ---
    cv::Mat m1(4, 4, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat m2(4, 4, CV_8UC3, cv::Scalar(200, 200, 200));
    Image<BGR> img1(m1), img2(m2);
    Image<BGR> blended = img1 | WeightedBlend<BGR>{img2}.alpha(0.5);
    std::cout << "WeightedBlend 50/50 mean: " << cv::mean(blended.mat())[0] << "\n"; // expect ~150

    // --- AlphaBlend ---
    cv::Mat bg_mat(4, 4, CV_8UC3,  cv::Scalar(0,   0,   0));
    cv::Mat ov_mat(4, 4, CV_8UC4,  cv::Scalar(200, 200, 200, 128)); // 50% alpha
    Image<BGR>  bg(bg_mat);
    Image<BGRA> overlay(ov_mat);
    Image<BGR>  composited = bg | AlphaBlend{overlay};
    std::cout << "AlphaBlend 50% overlay mean: " << cv::mean(composited.mat())[0] << "\n"; // expect ~100

    std::cout << "demo_color_adjust: OK\n";
    return 0;
}
