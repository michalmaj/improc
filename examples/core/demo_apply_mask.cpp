// examples/core/demo_apply_mask.cpp
//
// Demo: ApplyMask — zero pixels outside a binary Image<Gray> mask
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
    // ── Source image: gradient + shapes ───────────────────────────────────────
    cv::Mat raw(300, 400, CV_8UC3);
    for (int r = 0; r < raw.rows; ++r)
        for (int c = 0; c < raw.cols; ++c)
            raw.at<cv::Vec3b>(r, c) = {
                static_cast<uchar>(c * 255 / raw.cols),
                static_cast<uchar>(r * 255 / raw.rows),
                static_cast<uchar>(128)
            };
    cv::circle(raw, {200, 150}, 80, {0, 0, 255}, -1);
    cv::rectangle(raw, {20, 20}, {150, 100}, {255, 200, 0}, -1);

    Image<BGR> src(raw);
    std::cout << "Source: " << src.cols() << "x" << src.rows() << " BGR\n";
    show("0 — Source", src.mat());

    // ── 1. Circular mask ──────────────────────────────────────────────────────
    std::cout << "\n--- 1. Circular mask ---\n";

    cv::Mat circle_mask(raw.size(), CV_8UC1, cv::Scalar(0));
    cv::circle(circle_mask, {200, 150}, 100, cv::Scalar(255), -1);
    Image<Gray> mask_circle(circle_mask);

    Image<BGR> result_circle = src | ApplyMask{}.mask(mask_circle);
    show("1a — Source",               src.mat());
    show("1b — Circular mask",        circle_mask);
    show("1c — Applied circular mask", result_circle.mat());

    // ── 2. Rectangular mask ───────────────────────────────────────────────────
    std::cout << "\n--- 2. Rectangular mask ---\n";

    cv::Mat rect_mask(raw.size(), CV_8UC1, cv::Scalar(0));
    cv::rectangle(rect_mask, {80, 60}, {320, 240}, cv::Scalar(255), -1);
    Image<Gray> mask_rect(rect_mask);

    Image<BGR> result_rect = src | ApplyMask{}.mask(mask_rect);
    show("2a — Rectangular mask",        rect_mask);
    show("2b — Applied rectangular mask", result_rect.mat());

    // ── 3. Ring / donut mask ──────────────────────────────────────────────────
    std::cout << "\n--- 3. Ring mask ---\n";

    cv::Mat ring_mask(raw.size(), CV_8UC1, cv::Scalar(0));
    cv::circle(ring_mask, {200, 150}, 120, cv::Scalar(255), -1);
    cv::circle(ring_mask, {200, 150},  50, cv::Scalar(0),   -1);  // punch hole
    Image<Gray> mask_ring(ring_mask);

    Image<BGR> result_ring = src | ApplyMask{}.mask(mask_ring);
    show("3a — Ring mask",        ring_mask);
    show("3b — Applied ring mask", result_ring.mat());

    // ── 4. Gray image with mask ───────────────────────────────────────────────
    std::cout << "\n--- 4. Gray image ---\n";

    cv::Mat gray_raw;
    cv::cvtColor(raw, gray_raw, cv::COLOR_BGR2GRAY);
    Image<Gray> src_gray(gray_raw);

    Image<Gray> result_gray = src_gray | ApplyMask{}.mask(mask_circle);
    show("4a — Gray source",              src_gray.mat());
    show("4b — Gray + circular mask",     result_gray.mat());

    // ── 5. Pipeline: mask → blur masked region ────────────────────────────────
    std::cout << "\n--- 5. Pipeline: ApplyMask → GaussianBlur ---\n";

    Image<BGR> result_pipeline = src
        | ApplyMask{}.mask(mask_circle)
        | GaussianBlur{}.kernel_size(15).sigma(4.0);

    show("5 — Masked + blurred", result_pipeline.mat());

    std::cout << "Done.\n";
    return 0;
}
