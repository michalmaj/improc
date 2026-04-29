// examples/core/demo_geometric_preprocessing.cpp
// Demonstrates CenterCrop and LetterBox — geometric preprocessing ops.

#include <format>
#include <iostream>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

int main() {
    // Synthetic 480×640 BGR image (480 rows, 640 cols)
    cv::Mat mat(480, 640, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(mat);
    std::cout << std::format("Source: {}x{}\n", img.cols(), img.rows());

    // ── CenterCrop ──────────────────────────────────────────────────────────
    Image<BGR> cropped = img | CenterCrop{}.width(224).height(224);
    std::cout << std::format("CenterCrop(224x224): {}x{}\n",
        cropped.cols(), cropped.rows());

    cv::Mat gray_mat(480, 640, CV_8UC1);
    Image<Gray> gray_img(gray_mat);
    Image<Gray> gray_cropped = gray_img | CenterCrop{}.width(128).height(128);
    std::cout << std::format("CenterCrop Gray(128x128): {}x{}\n",
        gray_cropped.cols(), gray_cropped.rows());

    // ── LetterBox ────────────────────────────────────────────────────────────
    Image<BGR> lb640 = img | LetterBox{}.width(640).height(640);
    std::cout << std::format("LetterBox(640x640): {}x{}\n",
        lb640.cols(), lb640.rows());

    for (int sz : {416, 512, 640}) {
        Image<BGR> lb = img | LetterBox{}.width(sz).height(sz);
        std::cout << std::format("LetterBox({}x{}): {}x{}\n",
            sz, sz, lb.cols(), lb.rows());
    }

    Image<BGR> lb_black = img | LetterBox{}.width(224).height(224).value({0, 0, 0});
    std::cout << std::format("LetterBox black fill(224x224): {}x{}\n",
        lb_black.cols(), lb_black.rows());

    // ── Pipeline composition ─────────────────────────────────────────────────
    Image<Float32C3> tensor = img
        | LetterBox{}.width(256).height(256)
        | CenterCrop{}.width(224).height(224)
        | ToFloat32C3{}
        | NormalizeTo{0.0f, 1.0f};
    std::cout << std::format("Pipeline result: {}x{} Float32C3\n",
        tensor.cols(), tensor.rows());

    std::cout << "Done.\n";
    return 0;
}
