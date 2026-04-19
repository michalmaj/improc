// examples/core/demo_homography.cpp
//
// Demo: find_homography + WarpPerspective
//
// Simulates the sensor fusion scenario: a "camera image" (top-down photo)
// is warped onto a "sensor grid" (tomography reconstruction frame)
// using a homography computed from four corner correspondences.
//
// Usage: run from the build directory; press any key to advance.

#include "improc/core/pipeline.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace improc::core;

static void show(const std::string& title, const cv::Mat& mat) {
    cv::imshow(title, mat);
    cv::waitKey(0);
}

int main() {
    // ── Source: synthetic "camera image" (400×400) ────────────────────────────
    cv::Mat camera_raw(400, 400, CV_8UC3, cv::Scalar(30, 30, 80));
    cv::rectangle(camera_raw, {50, 50},   {150, 150}, {200, 200, 50},  -1);
    cv::circle(camera_raw,    {250, 200}, 60,          {50, 200, 200},  -1);
    cv::line(camera_raw,      {0, 350},   {400, 350},  {180, 180, 180},  3);
    Image<BGR> camera_img(camera_raw);
    show("0 — Camera image (simulated)", camera_img.mat());

    // ── Destination: synthetic "tomography frame" (300×300) ──────────────────
    cv::Mat tomo_raw(300, 300, CV_8UC3, cv::Scalar(10, 10, 10));
    cv::rectangle(tomo_raw, {10, 10}, {290, 290}, {60, 60, 60}, 2);
    Image<BGR> tomo_img(tomo_raw);
    show("1 — Tomography frame (simulated)", tomo_img.mat());

    // ── Correspondences: camera corners → tomo corners ────────────────────────
    std::vector<cv::Point2f> src_pts = {
        {0.f,   0.f},
        {400.f, 0.f},
        {400.f, 400.f},
        {0.f,   400.f}
    };
    std::vector<cv::Point2f> dst_pts = {
        {10.f,  10.f},
        {290.f, 10.f},
        {290.f, 290.f},
        {10.f,  290.f}
    };

    // ── 1. Compute homography ─────────────────────────────────────────────────
    std::cout << "\n--- find_homography ---\n";
    auto H_result = find_homography(src_pts, dst_pts);
    if (!H_result) {
        std::cerr << "Homography failed: " << H_result.error().message << "\n";
        return 1;
    }
    std::cout << "Homography matrix:\n" << *H_result << "\n";

    // ── 2. Warp camera image onto tomo frame size ────────────────────────────
    std::cout << "\n--- WarpPerspective (output = tomo size 300×300) ---\n";
    Image<BGR> warped = WarpPerspective{}
        .homography(*H_result)
        .width(300).height(300)(camera_img);
    show("2 — Warped camera onto tomo frame", warped.mat());

    // ── 3. Overlay: blend warped camera with tomo frame ──────────────────────
    std::cout << "\n--- Sensor fusion: alpha blend ---\n";
    cv::Mat blended;
    cv::addWeighted(tomo_raw, 0.5, warped.mat(), 0.5, 0.0, blended);
    show("3 — Fused (tomo 50% + warped camera 50%)", blended);

    // ── 4. Default size (same as input) ──────────────────────────────────────
    std::cout << "\n--- WarpPerspective (default size = input 400×400) ---\n";
    Image<BGR> warped_full = WarpPerspective{}.homography(*H_result)(camera_img);
    std::cout << "Output size: " << warped_full.cols() << "x" << warped_full.rows() << "\n";
    show("4 — Warped (default size, 400×400)", warped_full.mat());

    // ── 5. Pipeline form ──────────────────────────────────────────────────────
    std::cout << "\n--- Pipeline: camera → WarpPerspective → result ---\n";
    Image<BGR> pipeline_result = camera_img
        | WarpPerspective{}.homography(*H_result).width(300).height(300);
    show("5 — Pipeline warp (300×300)", pipeline_result.mat());

    // ── 6. Error case: insufficient points ───────────────────────────────────
    std::cout << "\n--- Error handling: 3 points (< 4 required) ---\n";
    std::vector<cv::Point2f> too_few = {{0,0},{100,0},{100,100}};
    auto bad = find_homography(too_few, too_few);
    if (!bad) {
        std::cout << "Expected error: " << bad.error().message << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
