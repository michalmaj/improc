//
// Created by Michał Maj on 09/04/2026.
//

#include "improc/core/pipeline.hpp"
#include "improc/core/convert.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace improc::core;

int main() {
    // --- 1. Create a sample BGR image ---
    cv::Mat raw(300, 400, CV_8UC3, cv::Scalar(60, 120, 200));
    Image<BGR> bgr(raw);
    std::cout << "Input:  Image<BGR>    type=" << bgr.mat().type()
              << "  size=" << bgr.cols() << "x" << bgr.rows() << "\n";

    cv::imshow("Step 0 — Input (BGR)", bgr.mat());
    cv::waitKey(0);

    // --- 2. Pipeline: BGR -> Gray -> Float32 ---
    Image<Float32> result = bgr | ToGray{} | ToFloat32{};
    std::cout << "Output: Image<Float32> type=" << result.mat().type()
              << "  size=" << result.cols() << "x" << result.rows() << "\n";

    cv::imshow("Step 2 — Output (Float32, normalized 0-1)", result.mat());
    cv::waitKey(0);

    // --- 3. Same result using free function convert<> ---
    Image<Gray> gray_free = convert<Gray, BGR>(bgr);
    std::cout << "convert<Gray,BGR> type=" << gray_free.mat().type() << "\n";

    cv::imshow("Step 1 — Intermediate (Gray via convert)", gray_free.mat());
    cv::waitKey(0);

    return 0;
}
