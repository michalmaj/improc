// examples/calib/demo_stereo.cpp
//
// Demo: StereoBM, StereoSGBM, ReprojectTo3D
//
// Creates a textured synthetic scene and a right image offset by a known
// horizontal baseline. Computes disparity with both methods and shows a
// side-by-side comparison. ReprojectTo3D demonstrates depth reconstruction.
// Press any key to advance.

#include "improc/calib/pipeline.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace improc::calib;
using namespace improc::core;

static void show(const std::string& t, const cv::Mat& m) {
    cv::imshow(t, m); cv::waitKey(0);
}

int main() {
    // ── Synthetic textured scene ──────────────────────────────────────────────
    const int W = 640, H = 480;
    cv::Mat scene(H, W, CV_8UC3, cv::Scalar(80, 80, 80));
    cv::rectangle(scene, {50,  50},  {200, 200}, {200,  80,  80}, -1);
    cv::rectangle(scene, {250, 100}, {350, 300}, { 80, 200,  80}, -1);
    cv::circle(scene,    {500, 240}, 80,          { 80,  80, 200}, -1);
    cv::rectangle(scene, {0,  400},  {640, 480},  {120, 120,  40}, -1);

    // Add texture via random noise
    cv::Mat noise(H, W, CV_8UC3);
    cv::randu(noise, 0, 30);
    scene += noise;

    Image<BGR> left_bgr(scene);
    Image<Gray> left_gray = left_bgr | ToGray{};

    // Right image: shift left by baseline_px (simulates positive disparity)
    const int baseline_px = 20;
    cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -baseline_px, 0, 1, 0);
    cv::Mat right_mat;
    cv::warpAffine(scene, right_mat, M, scene.size());
    Image<BGR> right_bgr(right_mat);
    Image<Gray> right_gray = right_bgr | ToGray{};

    show("1 — Left frame",  left_bgr.mat());
    show("2 — Right frame (shifted left by 20px)", right_bgr.mat());

    // ── StereoBM ──────────────────────────────────────────────────────────────
    std::cout << "\n--- StereoBM ---\n";
    cv::Mat disp_bm = StereoBM{}
        .num_disparities(64)   // disparity search range, must be divisible by 16 (default: 16)
        .block_size(15)        // matching block size, odd number 5-255 (default: 15)
        (left_gray, right_gray);   // returns CV_16S; divide by 16 for float disparity

    cv::Mat disp_bm_vis;
    disp_bm.convertTo(disp_bm_vis, CV_8U, 255.0 / (16.0 * 64));
    show("3 — StereoBM disparity", disp_bm_vis);

    // ── StereoSGBM ────────────────────────────────────────────────────────────
    std::cout << "\n--- StereoSGBM ---\n";
    const int bsz = 5;
    cv::Mat disp_sgbm = StereoSGBM{}
        .min_disparity(0)              // minimum disparity (default: 0)
        .num_disparities(64)           // disparity range (default: 64)
        .block_size(bsz)               // block size, must be odd (default: 3)
        .p1(8  * bsz * bsz)           // smoothness penalty 1 (default: 0)
        .p2(32 * bsz * bsz)           // smoothness penalty 2 (default: 0)
        .mode(cv::StereoSGBM::MODE_SGBM_3WAY)  // (default: MODE_SGBM)
        (left_gray, right_gray);       // returns CV_16S

    cv::Mat disp_sgbm_vis;
    disp_sgbm.convertTo(disp_sgbm_vis, CV_8U, 255.0 / (16.0 * 64));
    show("4 — StereoSGBM disparity (higher quality)", disp_sgbm_vis);

    // ── ReprojectTo3D ─────────────────────────────────────────────────────────
    std::cout << "\n--- ReprojectTo3D ---\n";
    // Synthetic Q matrix (normally from StereoRectify)
    double f = 500.0;
    cv::Mat Q = (cv::Mat_<double>(4, 4)
        << 1,  0,  0,   -W / 2.0,
           0,  1,  0,   -H / 2.0,
           0,  0,  0,    f,
           0,  0, 1.0 / 0.05, 0);

    cv::Mat pts3d = ReprojectTo3D{}
        .handle_missing(false)   // false = keep all points including invalid ones (default)
        (disp_sgbm, Q);          // returns CV_32FC3 point cloud

    int valid = 0;
    for (int y = 0; y < pts3d.rows; ++y)
        for (int x = 0; x < pts3d.cols; ++x)
            if (!std::isinf(pts3d.at<cv::Vec3f>(y, x)[2])) ++valid;
    std::cout << "Valid 3-D points: " << valid
              << " / " << pts3d.rows * pts3d.cols << "\n";

    std::cout << "Done.\n";
    return 0;
}
