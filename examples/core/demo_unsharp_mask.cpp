// examples/core/demo_unsharp_mask.cpp
//
// Demo: UnsharpMask — sharpening via blurred subtraction
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
    // ── Source image: synthetic scene with text and edges ─────────────────────
    cv::Mat raw(300, 400, CV_8UC3, cv::Scalar(180, 180, 180));
    cv::rectangle(raw, {30, 30},  {180, 130}, {60, 80, 200}, -1);
    cv::rectangle(raw, {220, 30}, {370, 130}, {200, 80, 60}, -1);
    cv::circle(raw, {200, 200}, 70, {50, 160, 50}, -1);
    cv::putText(raw, "Sharpness", {50, 270}, cv::FONT_HERSHEY_SIMPLEX, 0.9,
                {20, 20, 20}, 2);

    // Slightly blur the source to simulate a soft/out-of-focus image
    cv::GaussianBlur(raw, raw, cv::Size(0, 0), 1.2);

    Image<BGR> src(raw);
    std::cout << "Source: " << src.cols() << "x" << src.rows()
              << " BGR (pre-blurred to simulate soft image)\n";
    show("0 — Source (soft)", src.mat());

    // ── 1. Sigma sweep (fixed strength) ───────────────────────────────────────
    std::cout << "\n--- 1. Sigma sweep (strength=0.8) ---\n";

    Image<BGR> s_small  = src | UnsharpMask{}.sigma(0.5).strength(0.8);
    Image<BGR> s_medium = src | UnsharpMask{}.sigma(1.0).strength(0.8);
    Image<BGR> s_large  = src | UnsharpMask{}.sigma(2.0).strength(0.8);
    show("1a — sigma=0.5, strength=0.8 (fine detail)", s_small.mat());
    show("1b — sigma=1.0, strength=0.8 (default)",     s_medium.mat());
    show("1c — sigma=2.0, strength=0.8 (broad edges)", s_large.mat());

    // ── 2. Strength sweep (fixed sigma) ───────────────────────────────────────
    std::cout << "\n--- 2. Strength sweep (sigma=1.0) ---\n";

    Image<BGR> w_light  = src | UnsharpMask{}.sigma(1.0).strength(0.3);
    Image<BGR> w_normal = src | UnsharpMask{}.sigma(1.0).strength(1.0);
    Image<BGR> w_heavy  = src | UnsharpMask{}.sigma(1.0).strength(2.5);
    show("2a — sigma=1.0, strength=0.3 (subtle)",  w_light.mat());
    show("2b — sigma=1.0, strength=1.0 (moderate)", w_normal.mat());
    show("2c — sigma=1.0, strength=2.5 (heavy)",    w_heavy.mat());

    // ── 3. Gray image ─────────────────────────────────────────────────────────
    std::cout << "\n--- 3. Gray image ---\n";

    cv::Mat gray_raw;
    cv::cvtColor(raw, gray_raw, cv::COLOR_BGR2GRAY);
    Image<Gray> src_gray(gray_raw);

    Image<Gray> sharp_gray = src_gray | UnsharpMask{}.sigma(1.0).strength(1.5);
    show("3a — Gray source", src_gray.mat());
    show("3b — UnsharpMask Gray (sigma=1.0, strength=1.5)", sharp_gray.mat());

    // ── 4. Default parameters ─────────────────────────────────────────────────
    std::cout << "\n--- 4. Default parameters (sigma=1.0, strength=0.5) ---\n";

    Image<BGR> default_sharp = src | UnsharpMask{};
    show("4 — Default UnsharpMask", default_sharp.mat());

    // ── 5. Pipeline: blur → sharpen (deblur workflow) ─────────────────────────
    std::cout << "\n--- 5. Pipeline: extra blur → UnsharpMask ---\n";

    Image<BGR> result = src
        | GaussianBlur{}.kernel_size(5).sigma(1.5)  // simulate motion blur
        | UnsharpMask{}.sigma(1.5).strength(1.2);   // recover sharpness

    show("5a — Extra-blurred",          (src | GaussianBlur{}.kernel_size(5).sigma(1.5)).mat());
    show("5b — After UnsharpMask recovery", result.mat());

    std::cout << "Done.\n";
    return 0;
}
