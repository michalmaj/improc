// examples/core/demo_warp_affine.cpp
//
// Demo: WarpAffine — 2×3 affine transforms (translation, rotation, scaling, shear)
//
// Usage: run from the build directory; press any key to advance.

#include "improc/core/pipeline.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace improc::core;

static void show(const std::string& title, const cv::Mat& mat) {
    cv::imshow(title, mat);
    cv::waitKey(0);
}

int main() {
    // ── Source image: coloured rectangle with circle ───────────────────────────
    cv::Mat raw(300, 400, CV_8UC3, cv::Scalar(40, 40, 40));
    cv::rectangle(raw, {50, 50}, {350, 250}, {200, 100, 50}, -1);
    cv::circle(raw, {200, 150}, 60, {50, 200, 200}, -1);
    cv::putText(raw, "improc++", {100, 160}, cv::FONT_HERSHEY_SIMPLEX, 1.2,
                {255, 255, 255}, 2);

    Image<BGR> src(raw);
    std::cout << "Source: " << src.cols() << "x" << src.rows() << " BGR\n";
    show("0 — Source", src.mat());

    // ── 1. Translation ────────────────────────────────────────────────────────
    std::cout << "\n--- 1. Translation ---\n";

    cv::Mat T = cv::Mat::eye(2, 3, CV_64F);
    T.at<double>(0, 2) = 40.0;  // tx = 40px right
    T.at<double>(1, 2) = 30.0;  // ty = 30px down

    Image<BGR> translated = src | WarpAffine{}.matrix(T);
    show("1 — Translation (+40px right, +30px down)", translated.mat());
    std::cout << "  tx=40, ty=30\n";

    // ── 2. Rotation (via cv::getRotationMatrix2D) ─────────────────────────────
    std::cout << "\n--- 2. Rotation ---\n";

    cv::Point2f center(src.cols() / 2.0f, src.rows() / 2.0f);
    cv::Mat R15 = cv::getRotationMatrix2D(center, 15.0, 1.0);
    cv::Mat R45 = cv::getRotationMatrix2D(center, 45.0, 1.0);

    Image<BGR> rot15 = src | WarpAffine{}.matrix(R15);
    Image<BGR> rot45 = src | WarpAffine{}.matrix(R45);
    show("2a — Rotation 15°", rot15.mat());
    show("2b — Rotation 45°", rot45.mat());

    // ── 3. Scale + rotate combined ────────────────────────────────────────────
    std::cout << "\n--- 3. Scale + Rotate ---\n";

    cv::Mat RS = cv::getRotationMatrix2D(center, 20.0, 0.7);  // 70% scale, 20° rotate
    Image<BGR> scaled_rot = src | WarpAffine{}.matrix(RS);
    show("3 — Scale 0.7 + Rotate 20°", scaled_rot.mat());

    // ── 4. Shear ──────────────────────────────────────────────────────────────
    std::cout << "\n--- 4. Shear ---\n";

    cv::Mat Sh = cv::Mat::eye(2, 3, CV_64F);
    Sh.at<double>(0, 1) = 0.3;  // horizontal shear

    Image<BGR> sheared = src | WarpAffine{}.matrix(Sh);
    show("4 — Horizontal shear (0.3)", sheared.mat());

    // ── 5. Custom output size ─────────────────────────────────────────────────
    std::cout << "\n--- 5. Custom output size ---\n";

    cv::Mat I = cv::Mat::eye(2, 3, CV_64F);
    Image<BGR> cropped = src | WarpAffine{}.matrix(I).width(200).height(150);
    show("5 — Identity warp, output 200×150", cropped.mat());

    // ── 6. Gray image ─────────────────────────────────────────────────────────
    std::cout << "\n--- 6. Gray image ---\n";

    cv::Mat gray_raw;
    cv::cvtColor(raw, gray_raw, cv::COLOR_BGR2GRAY);
    Image<Gray> gray_src(gray_raw);

    Image<Gray> gray_rot = gray_src | WarpAffine{}.matrix(R15);
    show("6 — Rotation 15° on Gray image", gray_rot.mat());

    // ── 7. Pipeline composition ───────────────────────────────────────────────
    std::cout << "\n--- 7. Pipeline: Rotate → Translate ---\n";

    Image<BGR> result = src
        | WarpAffine{}.matrix(R15)
        | WarpAffine{}.matrix(T);

    show("7 — Rotate(15°) → Translate(+40,+30)", result.mat());

    std::cout << "Done.\n";
    return 0;
}
