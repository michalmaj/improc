//
// Created by Michał Maj on 15/04/2026.
//
// Demo: Blur, morphology, threshold, and padding ops.
//
// Usage: run from the build directory; press any key to advance between windows.

#include "improc/core/pipeline.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace improc::core;

static void show(const std::string& title, const cv::Mat& mat) {
    cv::imshow(title, mat);
    cv::waitKey(0);
}

int main() {
    // --- Source: noisy BGR gradient ---
    cv::Mat raw(300, 400, CV_8UC3);
    for (int r = 0; r < raw.rows; ++r)
        for (int c = 0; c < raw.cols; ++c)
            raw.at<cv::Vec3b>(r, c) = {
                static_cast<uchar>(c * 255 / raw.cols),
                static_cast<uchar>(r * 255 / raw.rows),
                static_cast<uchar>(128)
            };

    // Add some salt-and-pepper noise for the blur demo
    cv::RNG rng(42);
    cv::Mat noisy = raw.clone();
    for (int i = 0; i < 3000; ++i) {
        int r = rng.uniform(0, noisy.rows);
        int c = rng.uniform(0, noisy.cols);
        noisy.at<cv::Vec3b>(r, c) = (i % 2 == 0)
            ? cv::Vec3b{255, 255, 255}
            : cv::Vec3b{0, 0, 0};
    }

    Image<BGR> src(raw);
    Image<BGR> noisy_src(noisy);
    std::cout << "Source: " << src.cols() << "x" << src.rows() << " BGR\n";
    show("0 — Source (BGR)", src.mat());
    show("0b — Noisy source", noisy_src.mat());

    // -----------------------------------------------------------------------
    // 1. Blur ops
    // -----------------------------------------------------------------------
    std::cout << "\n--- Blur ---\n";

    Image<BGR> gaussian3  = noisy_src | GaussianBlur{}.kernel_size(3).sigma(1.0);
    Image<BGR> gaussian11 = noisy_src | GaussianBlur{}.kernel_size(11).sigma(2.0);
    show("1a — GaussianBlur (k=3, σ=1.0)", gaussian3.mat());
    show("1b — GaussianBlur (k=11, σ=2.0)", gaussian11.mat());

    Image<BGR> median3 = noisy_src | MedianBlur{}.kernel_size(3);
    Image<BGR> median7 = noisy_src | MedianBlur{}.kernel_size(7);
    show("1c — MedianBlur (k=3)", median3.mat());
    show("1d — MedianBlur (k=7) — stronger noise removal", median7.mat());

    // -----------------------------------------------------------------------
    // 2. Morphological ops (on grayscale for clarity)
    // -----------------------------------------------------------------------
    std::cout << "\n--- Morphology ---\n";

    // Build a binary-ish grayscale mask: white circle on black background
    cv::Mat mask_raw(200, 200, CV_8UC1, cv::Scalar(0));
    cv::circle(mask_raw, {100, 100}, 60, cv::Scalar(255), -1);
    // Add a few gaps/dots
    cv::rectangle(mask_raw, {70, 70}, {130, 130}, cv::Scalar(0), 8);
    Image<Gray> mask(mask_raw);
    show("2a — Morphology source (Gray mask)", mask.mat());

    Image<Gray> dilated_rect = mask | Dilate{}.kernel_size(9).shape(MorphShape::Rect);
    Image<Gray> dilated_ell  = mask | Dilate{}.kernel_size(9).shape(MorphShape::Ellipse);
    show("2b — Dilate (k=9, Rect)", dilated_rect.mat());
    show("2c — Dilate (k=9, Ellipse) — rounder result", dilated_ell.mat());

    Image<Gray> eroded = mask | Erode{}.kernel_size(7).iterations(2);
    show("2d — Erode (k=7, 2 iterations)", eroded.mat());

    // Opening = erode then dilate (removes small blobs)
    Image<Gray> opened = mask | Erode{}.kernel_size(5) | Dilate{}.kernel_size(5);
    show("2e — Opening (erode→dilate, k=5) — gap closure", opened.mat());

    // -----------------------------------------------------------------------
    // 3. Threshold ops (grayscale)
    // -----------------------------------------------------------------------
    std::cout << "\n--- Threshold ---\n";

    Image<Gray> gray = src | ToGray{};
    show("3a — Source (Gray)", gray.mat());

    Image<Gray> binary     = gray | Threshold{}.value(128).mode(ThresholdMode::Binary);
    Image<Gray> binary_inv = gray | Threshold{}.value(128).mode(ThresholdMode::BinaryInv);
    Image<Gray> truncate   = gray | Threshold{}.value(128).mode(ThresholdMode::Truncate);
    Image<Gray> to_zero    = gray | Threshold{}.value(128).mode(ThresholdMode::ToZero);
    show("3b — Binary (v=128)", binary.mat());
    show("3c — BinaryInv (v=128)", binary_inv.mat());
    show("3d — Truncate (v=128)", truncate.mat());
    show("3e — ToZero (v=128)", to_zero.mat());

    Image<Gray> otsu = gray | Threshold{}.mode(ThresholdMode::Otsu);
    std::cout << "  Otsu: threshold computed automatically from histogram\n";
    show("3f — Otsu (auto threshold)", otsu.mat());

    // -----------------------------------------------------------------------
    // 4. Padding ops
    // -----------------------------------------------------------------------
    std::cout << "\n--- Padding ---\n";

    // Symmetric constant pad — useful for adding border before convolution
    Image<BGR> pad_const = src
        | Pad{}.top(20).bottom(20).left(40).right(40)
               .mode(PadMode::Constant).value({114, 114, 114});
    std::cout << "  Pad Constant: " << pad_const.cols() << "x" << pad_const.rows() << "\n";
    show("4a — Pad Constant (gray border)", pad_const.mat());

    Image<BGR> pad_reflect = src
        | Pad{}.top(30).bottom(30).left(30).right(30)
               .mode(PadMode::Reflect);
    show("4b — Pad Reflect", pad_reflect.mat());

    Image<BGR> pad_replicate = src
        | Pad{}.top(30).bottom(30).left(30).right(30)
               .mode(PadMode::Replicate);
    show("4c — Pad Replicate", pad_replicate.mat());

    // PadToSquare: letterbox-style for inference pre-processing
    // Use a non-square source (400x300)
    Image<BGR> square = src | PadToSquare{}.value({114, 114, 114});
    std::cout << "  PadToSquare: " << src.cols() << "x" << src.rows()
              << " → " << square.cols() << "x" << square.rows() << "\n";
    show("4d — PadToSquare (letterbox, gray fill)", square.mat());

    // Pipeline: GaussianBlur → PadToSquare → Resize (typical inference prep)
    Image<BGR> prepped =
        noisy_src
        | GaussianBlur{}.kernel_size(5).sigma(1.5)
        | PadToSquare{}.value({114, 114, 114})
        | Resize{}.width(224).height(224);
    std::cout << "  Pipeline (blur→padtosquare→resize): "
              << prepped.cols() << "x" << prepped.rows() << "\n";
    show("4e — blur → PadToSquare → Resize(224)", prepped.mat());

    std::cout << "\nDone.\n";
    return 0;
}
