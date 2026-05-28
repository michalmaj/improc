// examples/core/demo_optical_flow.cpp
//
// Demo: SparseLKFlow (Lucas-Kanade), DenseFarnebackFlow, DenseDISFlow
//
// Creates two synthetic frames shifted by (20, 10) px, tracks with all three
// methods, prints flow estimates, shows dense flow as an HSV colour map.
// Press any key to advance.

#include "improc/core/pipeline.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace improc::core;

static void show(const std::string& title, const cv::Mat& m) {
    cv::imshow(title, m);
    cv::waitKey(0);
}

static cv::Mat flow_to_hsv(const Image<Flow>& flow) {
    cv::Mat planes[2];
    cv::split(flow.mat(), planes);
    cv::Mat mag, ang;
    cv::cartToPolar(planes[0], planes[1], mag, ang, true);
    cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
    mag.convertTo(mag, CV_8U);
    cv::Mat hue; ang.convertTo(hue, CV_8U, 255.0 / 360.0);
    cv::Mat hsv, bgr;
    cv::merge(std::vector<cv::Mat>{hue, cv::Mat::ones(mag.size(), CV_8U) * 255, mag}, hsv);
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    return bgr;
}

int main() {
    // ── Synthetic frame pair: frame2 = frame1 shifted right 20px, down 10px ──
    cv::Mat raw(240, 320, CV_8UC3, cv::Scalar(40, 40, 40));
    cv::rectangle(raw, {60,  60},  {140, 140}, {200, 100,  50}, -1);
    cv::circle(raw,    {220, 120}, 45,          { 50, 150, 200}, -1);

    const double gt_dx = 20.0, gt_dy = 10.0;
    cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, gt_dx, 0, 1, gt_dy);
    cv::Mat raw2;
    cv::warpAffine(raw, raw2, M, raw.size());

    Image<BGR>  bgr1(raw), bgr2(raw2);
    Image<Gray> gray1 = bgr1 | ToGray{};
    Image<Gray> gray2 = bgr2 | ToGray{};

    show("1 — frame1", bgr1.mat());
    show("2 — frame2 (shifted +20px right, +10px down)", bgr2.mat());

    // ── Sparse Lucas-Kanade ───────────────────────────────────────────────────
    std::cout << "\n--- SparseLKFlow ---\n";
    std::vector<cv::Point2f> pts1;
    cv::goodFeaturesToTrack(gray1.mat(), pts1, 100, 0.01, 10);

    SparseLKFlowResult lk = SparseLKFlow{}
        .win_size({21, 21})  // search window per pyramid level (default: {21,21})
        .max_level(3)        // pyramid depth (default: 3)
        .max_iter(30)        // max LK iterations (default: 30)
        .epsilon(0.01)       // convergence threshold (default: 0.01)
        (gray1, gray2, pts1);

    double sum_dx = 0, sum_dy = 0;
    int tracked = 0;
    for (std::size_t i = 0; i < lk.status.size(); ++i) {
        if (!lk.status[i]) continue;
        sum_dx += lk.points[i].x - pts1[i].x;
        sum_dy += lk.points[i].y - pts1[i].y;
        ++tracked;
    }
    if (tracked)
        std::cout << "Avg shift: dx=" << sum_dx / tracked
                  << "  dy=" << sum_dy / tracked
                  << "  (expected dx=" << gt_dx << ", dy=" << gt_dy << ")\n";

    // ── Dense Farneback ───────────────────────────────────────────────────────
    std::cout << "\n--- DenseFarnebackFlow ---\n";
    Image<Flow> farne = DenseFarnebackFlow{}
        .pyr_scale(0.5)    // pyramid scale per level (default: 0.5)
        .levels(3)         // pyramid depth (default: 3)
        .win_size(15)      // averaging window size (default: 15)
        .iterations(3)     // iterations per level (default: 3)
        .poly_n(5)         // polynomial neighbourhood (default: 5)
        .poly_sigma(1.2)   // Gaussian sigma for poly expansion (default: 1.2)
        (gray1, gray2);
    cv::Scalar fmean = cv::mean(farne.mat());
    std::cout << "Mean flow: dx=" << fmean[0] << "  dy=" << fmean[1] << "\n";
    show("3 — Farneback flow (HSV colour wheel)", flow_to_hsv(farne));

    // ── Dense DIS ─────────────────────────────────────────────────────────────
    std::cout << "\n--- DenseDISFlow (Fast preset) ---\n";
    Image<Flow> dis = DenseDISFlow{}
        .preset(DenseDISFlow::Preset::Fast)  // balanced speed/quality (default)
        (gray1, gray2);
    cv::Scalar dmean = cv::mean(dis.mat());
    std::cout << "DIS mean flow: dx=" << dmean[0] << "  dy=" << dmean[1] << "\n";
    show("4 — DIS flow (HSV colour wheel)", flow_to_hsv(dis));

    std::cout << "Done.\n";
    return 0;
}
