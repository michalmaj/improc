// examples/core/demo_color_lab_ycrcb.cpp
//
// Demo: ToLAB, ToYCrCb, ToBGR round-trips

#include <format>
#include <iostream>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

int main() {
    cv::Mat raw(4, 4, CV_8UC3, cv::Scalar(100, 150, 200)); // BGR (100, 150, 200)
    Image<BGR> bgr(raw);

    // ── ToLAB ────────────────────────────────────────────────────────────────
    Image<LAB> lab = bgr | ToLAB{};
    Image<BGR> from_lab = lab | ToBGR{};
    cv::Mat diff_lab;
    cv::absdiff(bgr.mat(), from_lab.mat(), diff_lab);
    cv::Mat diff_lab_gray;
    cv::cvtColor(diff_lab, diff_lab_gray, cv::COLOR_BGR2GRAY);
    std::cout << std::format("LAB round-trip max error: {:.1f}\n", cv::norm(diff_lab_gray, cv::NORM_INF));

    // ── ToYCrCb ───────────────────────────────────────────────────────────────
    Image<YCrCb> ycrcb = bgr | ToYCrCb{};
    Image<BGR> from_ycrcb = ycrcb | ToBGR{};
    cv::Mat diff_ycrcb;
    cv::absdiff(bgr.mat(), from_ycrcb.mat(), diff_ycrcb);
    cv::Mat diff_ycrcb_gray;
    cv::cvtColor(diff_ycrcb, diff_ycrcb_gray, cv::COLOR_BGR2GRAY);
    std::cout << std::format("YCrCb round-trip max error: {:.1f}\n", cv::norm(diff_ycrcb_gray, cv::NORM_INF));

    std::cout << "demo_color_lab_ycrcb: OK\n";
    return 0;
}
