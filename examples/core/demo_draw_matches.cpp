// examples/core/demo_draw_matches.cpp
//
// Demo: DrawKeypoints, DrawMatches
//
// Usage: run from the build directory; press any key to advance windows.

#include "improc/core/pipeline.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <format>
#include <iostream>

using namespace improc::core;

static void show(const std::string& title, const cv::Mat& mat) {
    cv::imshow(title, mat);
    cv::waitKey(0);
}

int main() {
    cv::Mat raw(300, 400, CV_8UC3, cv::Scalar(80, 80, 80));
    cv::rectangle(raw, {20,  20},  {150, 150}, cv::Scalar(220, 220, 220), -1);
    cv::rectangle(raw, {180, 20},  {360, 100}, cv::Scalar(30,  30,  30),  -1);
    cv::circle(raw,    {100, 230}, 60,          cv::Scalar(200, 160, 80),  -1);
    cv::ellipse(raw,   {300, 220}, {70, 40}, 30, 0, 360, cv::Scalar(50, 100, 200), -1);

    Image<BGR>  bgr(raw);
    Image<Gray> gray = bgr | ToGray{};

    show("0 — Source BGR", bgr.mat());

    // ── DrawKeypoints: ORB ────────────────────────────────────────────────────
    KeypointSet kps_orb = gray | DetectORB{}.max_features(200);
    Image<BGR>  orb_vis = gray | DrawKeypoints{kps_orb};
    std::cout << std::format("ORB: {} keypoints\n", kps_orb.size());
    show("1a — DrawKeypoints (ORB, Gray input)", orb_vis.mat());

    Image<BGR> orb_vis_bgr = bgr | DrawKeypoints{kps_orb};
    show("1b — DrawKeypoints (ORB, BGR input)", orb_vis_bgr.mat());

    // ── DrawKeypoints: SIFT ───────────────────────────────────────────────────
    KeypointSet kps_sift = gray | DetectSIFT{};
    Image<BGR>  sift_vis = gray | DrawKeypoints{kps_sift};
    std::cout << std::format("SIFT: {} keypoints\n", kps_sift.size());
    show("2 — DrawKeypoints (SIFT)", sift_vis.mat());

    // ── DrawMatches: self-match ────────────────────────────────────────────────
    DescriptorSet desc_orb  = gray | DescribeORB{kps_orb};
    MatchSet ms_bf           = MatchBF{desc_orb, desc_orb}();
    Image<BGR> match_bf_vis  = DrawMatches{bgr, kps_orb, bgr, kps_orb, ms_bf}();
    std::cout << std::format("ORB BF matches: {} — output size: {}x{}\n",
        ms_bf.size(), match_bf_vis.mat().cols, match_bf_vis.mat().rows);
    show("3 — DrawMatches (ORB BF, self-match)", match_bf_vis.mat());

    // ── DrawMatches: SIFT FLANN ───────────────────────────────────────────────
    DescriptorSet desc_sift = gray | DescribeSIFT{kps_sift};
    MatchSet ms_flann        = MatchFlann{desc_sift, desc_sift}();
    Image<BGR> match_fl_vis  = DrawMatches{bgr, kps_sift, bgr, kps_sift, ms_flann}();
    std::cout << std::format("SIFT FLANN matches: {} — output size: {}x{}\n",
        ms_flann.size(), match_fl_vis.mat().cols, match_fl_vis.mat().rows);
    show("4 — DrawMatches (SIFT FLANN, self-match)", match_fl_vis.mat());

    // ── Empty MatchSet ────────────────────────────────────────────────────────
    Image<BGR> empty_vis = DrawMatches{bgr, kps_orb, bgr, kps_orb, MatchSet{}}();
    std::cout << std::format("Empty matches: output {}x{}\n",
        empty_vis.mat().cols, empty_vis.mat().rows);
    show("5 — DrawMatches (empty MatchSet)", empty_vis.mat());

    std::cout << "Done.\n";
    return 0;
}
