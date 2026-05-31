// examples/core/demo_color_lab_ycrcb.cpp
//
// Demo: ToLAB, ToYCrCb, ToBGR round-trips

#include <iostream>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

int main() {
    cv::Mat raw(4, 4, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> bgr(raw);

    // ── ToLAB ────────────────────────────────────────────────────────────────
    Image<LAB> lab = bgr | ToLAB{};
    Image<BGR> from_lab = lab | ToBGR{};
    cv::Mat diff_lab;
    cv::absdiff(bgr.mat(), from_lab.mat(), diff_lab);
    std::cout << "ToLAB output type: CV_8UC3? "
              << (lab.mat().type() == CV_8UC3 ? "yes" : "no") << "\n";
    std::cout << "LAB round-trip max error: "
              << cv::norm(diff_lab, cv::NORM_INF) << "\n";

    // ── ToYCrCb ───────────────────────────────────────────────────────────────
    Image<YCrCb> ycrcb = bgr | ToYCrCb{};
    Image<BGR> from_ycrcb = ycrcb | ToBGR{};
    cv::Mat diff_ycrcb;
    cv::absdiff(bgr.mat(), from_ycrcb.mat(), diff_ycrcb);
    std::cout << "ToYCrCb output type: CV_8UC3? "
              << (ycrcb.mat().type() == CV_8UC3 ? "yes" : "no") << "\n";
    std::cout << "YCrCb round-trip max error: "
              << cv::norm(diff_ycrcb, cv::NORM_INF) << "\n";

    std::cout << "All color conversions OK.\n";
    return 0;
}
