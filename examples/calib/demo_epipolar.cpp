// examples/calib/demo_epipolar.cpp
//
// Demo: FindFundamentalMat, FindEssentialMat, RecoverPose, TriangulatePoints
//
// Uses synthetic point correspondences from a frontoparallel scene with a
// known horizontal baseline. No physical camera or image file required.

#include <format>
#include <iostream>
#include "improc/calib/pipeline.hpp"

using namespace improc::calib;
using namespace improc::core;

int main() {
    // Shared camera intrinsics (synthetic 640x480 camera)
    cv::Mat K = (cv::Mat_<double>(3,3) <<
        400., 0., 320.,
        0., 400., 240.,
        0., 0.,   1.);

    // Synthetic correspondences: N points, projected from two views offset
    // by a 30-pixel horizontal translation (pure translation in x).
    const int N = 30;
    std::vector<cv::Point2f> pts1, pts2;
    cv::RNG rng(42);
    for (int i = 0; i < N; ++i) {
        float x = rng.uniform(100.f, 500.f);
        float y = rng.uniform(80.f,  400.f);
        pts1.push_back({x, y});
        pts2.push_back({x - 30.f, y});
    }

    // ── FindFundamentalMat ────────────────────────────────────────────────────
    auto f_res = FindFundamentalMat{}(pts1, pts2);
    std::cout << std::format("F matrix: {}x{}\n", f_res.F.rows, f_res.F.cols);
    std::cout << std::format("F inliers: {}/{}\n",
                             cv::countNonZero(f_res.mask), N);

    // ── FindEssentialMat ──────────────────────────────────────────────────────
    auto e_res = FindEssentialMat{}(pts1, pts2, K);
    std::cout << std::format("E matrix: {}x{}\n", e_res.E.rows, e_res.E.cols);
    std::cout << std::format("E inliers: {}/{}\n",
                             cv::countNonZero(e_res.mask), N);

    // ── RecoverPose ───────────────────────────────────────────────────────────
    auto pose = RecoverPose{}(e_res.E, pts1, pts2, K);
    std::cout << std::format("RecoverPose R: {}x{}  t: {}x{}  inliers={}\n",
                             pose.R.rows, pose.R.cols,
                             pose.t.rows, pose.t.cols,
                             pose.inliers);

    // ── TriangulatePoints ─────────────────────────────────────────────────────
    cv::Mat P1(3, 4, CV_64F, cv::Scalar(0));
    cv::Mat(K * cv::Mat::eye(3, 3, CV_64F)).copyTo(P1(cv::Rect(0, 0, 3, 3)));

    cv::Mat Rt(3, 4, CV_64F, cv::Scalar(0));
    pose.R.copyTo(Rt(cv::Rect(0, 0, 3, 3)));
    pose.t.copyTo(Rt.col(3));
    cv::Mat P2 = K * Rt;

    cv::Mat pts4d = TriangulatePoints{}(P1, P2, pts1, pts2);
    std::cout << std::format("Triangulated {} 3-D points (homogeneous 4xN)\n",
                             pts4d.cols);

    std::cout << "demo_epipolar: OK\n";
    return 0;
}
