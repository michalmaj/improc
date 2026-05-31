// examples/core/demo_filter_ops.cpp
//
// Demo: BilateralFilter, CenterCrop, Remap

#include <format>
#include <iostream>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

int main() {
    // ── BilateralFilter ───────────────────────────────────────────────────────
    cv::Mat noisy(100, 100, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat noise(100, 100, CV_8UC3);
    cv::randn(noise, 0, 15);
    noisy += noise;
    Image<BGR> noisy_img(noisy);

    Image<BGR> smooth = noisy_img
        | BilateralFilter{}.diameter(9).sigma_color(75).sigma_space(75);

    std::cout << std::format("BilateralFilter output size: {}x{}\n",
                             smooth.cols(), smooth.rows());

    // ── CenterCrop ────────────────────────────────────────────────────────────
    Image<BGR> crop = noisy_img | CenterCrop{}.width(64).height(64);
    std::cout << std::format("CenterCrop 64x64 from 100x100: {}x{}\n",
                             crop.cols(), crop.rows());

    // ── Remap — horizontal mirror ──────────────────────────────────────────────
    cv::Mat map1(100, 100, CV_32FC1);
    cv::Mat map2(100, 100, CV_32FC1);
    for (int y = 0; y < 100; ++y)
        for (int x = 0; x < 100; ++x) {
            map1.at<float>(y, x) = static_cast<float>(99 - x);
            map2.at<float>(y, x) = static_cast<float>(y);
        }
    Image<BGR> mirrored = noisy_img | Remap(map1, map2);
    std::cout << std::format("Remap (mirror) output size: {}x{}\n",
                             mirrored.cols(), mirrored.rows());

    auto px_in  = noisy_img.mat().at<cv::Vec3b>(10, 20);
    auto px_out = mirrored.mat().at<cv::Vec3b>(10, 79);
    std::cout << std::format("Mirror check B channel: in={} out={} match={}\n",
                             (int)px_in[0], (int)px_out[0],
                             px_in[0] == px_out[0] ? "yes" : "no");

    std::cout << "demo_filter_ops: OK\n";
    return 0;
}
