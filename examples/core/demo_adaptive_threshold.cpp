// examples/core/demo_adaptive_threshold.cpp
// Demonstrates AdaptiveThreshold — locally-computed image binarisation.

#include <format>
#include <iostream>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

int main() {
    // Synthetic 100×100 grayscale image: dark gradient left-to-right
    cv::Mat mat(100, 100, CV_8UC1);
    for (int c = 0; c < 100; ++c)
        mat.col(c) = cv::Scalar(static_cast<uchar>(c * 2));
    Image<Gray> img{mat};
    std::cout << std::format("Source: {}x{} Gray\n", img.cols(), img.rows());

    // ── Default (Gaussian, Binary, block_size=11, C=2) ──────────────────────
    Image<Gray> result = img | AdaptiveThreshold{};
    std::cout << std::format("Default (Gaussian, block=11, C=2): {}x{}\n",
        result.cols(), result.rows());

    // ── Mean method ─────────────────────────────────────────────────────────
    Image<Gray> mean_result = img
        | AdaptiveThreshold{}.method(AdaptiveMethod::Mean).block_size(11).C(2);
    std::cout << std::format("Mean method (block=11, C=2): {}x{}\n",
        mean_result.cols(), mean_result.rows());

    // ── Inverted output ──────────────────────────────────────────────────────
    Image<Gray> inverted = img | AdaptiveThreshold{}.block_size(11).C(2).invert();
    std::cout << std::format("Inverted (BinaryInv): {}x{}\n",
        inverted.cols(), inverted.rows());

    // ── Custom parameters ────────────────────────────────────────────────────
    Image<Gray> custom = img
        | AdaptiveThreshold{}.block_size(31).C(5).max_value(200);
    std::cout << std::format("Custom (block=31, C=5, max=200): {}x{}\n",
        custom.cols(), custom.rows());

    // ── Pipeline composition ─────────────────────────────────────────────────
    // Typical document preprocessing: blur first to reduce noise, then threshold
    Image<Gray> pipeline_result = img
        | GaussianBlur{}.kernel_size(3)
        | AdaptiveThreshold{}.block_size(11).C(2);
    std::cout << std::format("GaussianBlur | AdaptiveThreshold: {}x{}\n",
        pipeline_result.cols(), pipeline_result.rows());

    std::cout << "Done.\n";
    return 0;
}
