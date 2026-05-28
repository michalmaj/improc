// examples/calib/demo_pose.cpp
//
// Demo: SolvePnP, SolvePnPRansac, ProjectPoints
//
// Defines a 3-D cube's corners, projects them with a known K and pose,
// then recovers the pose via SolvePnP and re-projects to verify round-trip.
// Press any key to close.

#include "improc/calib/pipeline.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace improc::calib;

int main() {
    // ── Synthetic K for a 640×480 image ──────────────────────────────────────
    cv::Mat K = (cv::Mat_<double>(3, 3)
        << 500, 0, 320,
           0, 500, 240,
           0,   0,   1);
    cv::Mat dist = cv::Mat::zeros(5, 1, CV_64F);

    // ── 3-D object: front + back face of a 1 m cube ───────────────────────────
    std::vector<cv::Point3f> obj_pts = {
        {-0.5f, -0.5f, 0.f}, { 0.5f, -0.5f, 0.f},
        { 0.5f,  0.5f, 0.f}, {-0.5f,  0.5f, 0.f},
        {-0.5f, -0.5f, 1.f}, { 0.5f, -0.5f, 1.f},
        { 0.5f,  0.5f, 1.f}, {-0.5f,  0.5f, 1.f}
    };

    // ── Ground-truth pose: ~10° rotation around Y, 3 m along Z ───────────────
    cv::Mat rvec_gt = (cv::Mat_<double>(3, 1) << 0, 0.174533, 0);
    cv::Mat tvec_gt = (cv::Mat_<double>(3, 1) << 0, 0, 3.0);

    // ── Project 3-D → 2-D with known pose ────────────────────────────────────
    std::vector<cv::Point2f> img_pts = ProjectPoints{}(obj_pts, rvec_gt, tvec_gt, K, dist);
    std::cout << "Projected " << img_pts.size() << " points\n";

    // ── SolvePnP: recover pose from 2-D/3-D correspondences ──────────────────
    PnPResult res = SolvePnP{}
        .method(cv::SOLVEPNP_ITERATIVE)   // default method
        (obj_pts, img_pts, K, dist);

    if (!res.success) { std::cerr << "SolvePnP failed\n"; return 1; }
    std::cout << "Recovered rvec: " << res.rvec.t() << "\n";
    std::cout << "Expected  rvec: " << rvec_gt.t()  << "\n";
    std::cout << "Recovered tvec: " << res.tvec.t() << "\n";
    std::cout << "Expected  tvec: " << tvec_gt.t()  << "\n";

    // ── Re-project with recovered pose (should match img_pts) ─────────────────
    auto reproj = ProjectPoints{}(obj_pts, res.rvec, res.tvec, K, dist);
    double max_err = 0;
    for (std::size_t i = 0; i < img_pts.size(); ++i)
        max_err = std::max(max_err, (double)cv::norm(img_pts[i] - reproj[i]));
    std::cout << "Max re-projection error: " << max_err << " px\n";

    // ── SolvePnPRansac: robust to outliers ────────────────────────────────────
    PnPRansacResult rr = SolvePnPRansac{}
        .method(cv::SOLVEPNP_ITERATIVE)
        .confidence(0.99)           // desired probability (default: 0.99)
        .reprojection_error(8.0f)   // inlier threshold in px (default: 8.0)
        .iterations(100)            // RANSAC iterations (default: 100)
        (obj_pts, img_pts, K, dist);
    std::cout << "RANSAC success: " << rr.success
              << "  inliers: " << cv::countNonZero(rr.inliers) << "\n";

    // ── Visualise projected cube ──────────────────────────────────────────────
    cv::Mat canvas(480, 640, CV_8UC3, cv::Scalar(30, 30, 30));
    const std::vector<std::pair<int,int>> edges = {
        {0,1},{1,2},{2,3},{3,0},
        {4,5},{5,6},{6,7},{7,4},
        {0,4},{1,5},{2,6},{3,7}
    };
    for (auto [a, b] : edges)
        cv::line(canvas, reproj[a], reproj[b], {0, 255, 0}, 2);

    cv::imshow("Projected cube (recovered pose)", canvas);
    cv::waitKey(0);

    std::cout << "Done.\n";
    return 0;
}
