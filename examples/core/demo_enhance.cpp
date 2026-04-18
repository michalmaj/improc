// examples/core/demo_enhance.cpp
//
// Demo: CLAHE, GammaCorrection, BilateralFilter
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
    // ── Source images ──────────────────────────────────────────────────────────
    // Low-contrast grayscale ramp (simulate a dark, flat image)
    cv::Mat gray_raw(300, 400, CV_8UC1);
    for (int r = 0; r < gray_raw.rows; ++r)
        for (int c = 0; c < gray_raw.cols; ++c)
            gray_raw.at<uchar>(r, c) = static_cast<uchar>(60 + (c * 80) / gray_raw.cols);

    // BGR gradient with a circle overlay (for bilateral + gamma)
    cv::Mat bgr_raw(300, 400, CV_8UC3);
    for (int r = 0; r < bgr_raw.rows; ++r)
        for (int c = 0; c < bgr_raw.cols; ++c)
            bgr_raw.at<cv::Vec3b>(r, c) = {
                static_cast<uchar>(c * 255 / bgr_raw.cols),
                static_cast<uchar>(r * 255 / bgr_raw.rows),
                static_cast<uchar>(128)
            };
    cv::circle(bgr_raw, {200, 150}, 80, {0, 0, 255}, -1);

    Image<Gray> gray(gray_raw);
    Image<BGR>  bgr(bgr_raw);

    std::cout << "Source: " << bgr.cols() << "x" << bgr.rows() << " BGR\n";
    show("0a — Source (Gray, low contrast)", gray.mat());
    show("0b — Source (BGR)", bgr.mat());

    // ── 1. CLAHE ──────────────────────────────────────────────────────────────
    std::cout << "\n--- CLAHE ---\n";

    Image<Gray> clahe_gray = gray | CLAHE{};
    Image<Gray> clahe_strong = gray | CLAHE{}.clip_limit(2.0).tile_grid_size(4, 4);
    show("1a — CLAHE Gray (clip=40, tile=8x8)", clahe_gray.mat());
    show("1b — CLAHE Gray (clip=2.0, tile=4x4) — finer local contrast", clahe_strong.mat());

    Image<BGR> clahe_bgr = bgr | CLAHE{}.clip_limit(3.0);
    show("1c — CLAHE BGR (colour-safe via LAB L-channel)", clahe_bgr.mat());

    // ── 2. GammaCorrection ────────────────────────────────────────────────────
    std::cout << "\n--- GammaCorrection ---\n";

    Image<BGR> gamma_bright = bgr | GammaCorrection{}.gamma(0.4f);
    Image<BGR> gamma_dark   = bgr | GammaCorrection{}.gamma(2.5f);
    Image<BGR> gamma_linear = bgr | GammaCorrection{}.gamma(1.0f);
    show("2a — Gamma 0.4 (brightened)", gamma_bright.mat());
    show("2b — Gamma 2.5 (darkened)", gamma_dark.mat());
    show("2c — Gamma 1.0 (identity)", gamma_linear.mat());

    Image<Gray> gamma_gray_bright = gray | GammaCorrection{}.gamma(0.5f);
    show("2d — GammaCorrection Gray (gamma=0.5)", gamma_gray_bright.mat());

    std::cout << "  gamma < 1 brightens; gamma > 1 darkens; gamma = 1 is identity\n";

    // ── 3. BilateralFilter ────────────────────────────────────────────────────
    std::cout << "\n--- BilateralFilter ---\n";

    // Add noise to compare filtering results
    cv::Mat noisy = bgr_raw.clone();
    cv::RNG rng(42);
    for (int i = 0; i < 2000; ++i) {
        int r = rng.uniform(0, noisy.rows);
        int c = rng.uniform(0, noisy.cols);
        noisy.at<cv::Vec3b>(r, c) = (i % 2 == 0)
            ? cv::Vec3b{255, 255, 255} : cv::Vec3b{0, 0, 0};
    }
    Image<BGR> noisy_src(noisy);
    show("3a — Noisy source", noisy_src.mat());

    Image<BGR> bilateral  = noisy_src | BilateralFilter{};
    Image<BGR> bilateral2 = noisy_src | BilateralFilter{}.diameter(15).sigma_color(80).sigma_space(80);
    Image<BGR> gaussian_ref = noisy_src | GaussianBlur{}.kernel_size(9).sigma(2.0);
    show("3b — BilateralFilter (d=9, σ=75) — edges preserved", bilateral.mat());
    show("3c — BilateralFilter (d=15, σ=80) — stronger, still sharp at edges", bilateral2.mat());
    show("3d — GaussianBlur (k=9) for comparison — edges blurred", gaussian_ref.mat());

    Image<Gray> bilat_gray = (gray | GammaCorrection{}.gamma(0.5f))
                           | BilateralFilter{}.diameter(9);
    show("3e — BilateralFilter Gray", bilat_gray.mat());

    // ── Combined pipeline ─────────────────────────────────────────────────────
    std::cout << "\n--- Combined pipeline ---\n";

    Image<BGR> result = bgr
        | GammaCorrection{}.gamma(0.6f)
        | BilateralFilter{}.diameter(9).sigma_color(60).sigma_space(60)
        | CLAHE{}.clip_limit(2.0);

    show("4  — Gamma(0.6) → Bilateral → CLAHE", result.mat());
    std::cout << "Done.\n";
    return 0;
}
