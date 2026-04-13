//
// Created by Michał Maj on 13/04/2026.
//

#include "improc/visualization/visualization.hpp"
#include "improc/core/pipeline.hpp"
#include <iostream>
#include <vector>

using namespace improc::core;
using namespace improc::visualization;

int main() {
    // --- 1. BGR histogram ---
    cv::Mat bgr_mat(200, 300, CV_8UC3);
    cv::randu(bgr_mat, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    Image<BGR> bgr(bgr_mat);
    std::cout << "BGR histogram (512x256, 256 bins)...\n";
    bgr | Histogram{} | Show{"BGR Histogram"};

    // --- 2. Gray histogram with custom params ---
    cv::Mat gray_mat(200, 300, CV_8UC1);
    cv::randu(gray_mat, cv::Scalar(0), cv::Scalar(255));
    Image<Gray> gray(gray_mat);
    std::cout << "Gray histogram (400x200, 64 bins)...\n";
    gray | Histogram{}.bins(64).width(400).height(200) | Show{"Gray Histogram"};

    // --- 3. Float32 histogram ---
    cv::Mat f32_mat(100, 100, CV_32FC1);
    cv::randu(f32_mat, cv::Scalar(0.0f), cv::Scalar(1.0f));
    Image<Float32> f32(f32_mat);
    std::cout << "Float32 histogram (auto range)...\n";
    f32 | Histogram{} | Show{"Float32 Histogram"};

    // --- 4. Line plot ---
    std::vector<float> loss = {1.2f, 0.95f, 0.78f, 0.62f, 0.50f, 0.41f, 0.35f, 0.30f};
    std::cout << "Line plot (training loss)...\n";
    LinePlot{}.title("Train Loss").color({0, 200, 255})(loss) | Show{"Train Loss"};

    // --- 5. Scatter plot ---
    std::vector<float> xs = {0.1f, 0.4f, 0.7f, 0.2f, 0.9f, 0.5f, 0.3f, 0.8f};
    std::vector<float> ys = {0.3f, 0.6f, 0.2f, 0.8f, 0.5f, 0.1f, 0.7f, 0.4f};
    std::cout << "Scatter plot...\n";
    Scatter{}.title("Features").color({0, 255, 255}).point_radius(5)(xs, ys)
        | Show{"Scatter"};

    std::cout << "Done.\n";
    return 0;
}
