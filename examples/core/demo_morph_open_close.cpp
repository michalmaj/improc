// examples/core/demo_morph_open_close.cpp
// Demonstrates MorphOpen (removes noise) and MorphClose (fills holes).

#include <format>
#include <iostream>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

int main() {
    // ── MorphOpen: remove isolated bright pixel (noise) ──────────────────────
    cv::Mat noisy_mat(20, 20, CV_8UC1, cv::Scalar(0));
    noisy_mat.at<uchar>(10, 10) = 255;    // single bright pixel
    Image<Gray> noisy{noisy_mat};
    Image<Gray> opened = noisy | MorphOpen{}.kernel_size(3);
    std::cout << std::format("MorphOpen: noise pixel after open = {} (was 255)\n",
        (int)opened.mat().at<uchar>(10, 10));  // 0 — removed

    // ── MorphClose: fill isolated dark pixel (hole) ──────────────────────────
    cv::Mat holed_mat(20, 20, CV_8UC1, cv::Scalar(255));
    holed_mat.at<uchar>(10, 10) = 0;      // single dark hole
    Image<Gray> holed{holed_mat};
    Image<Gray> closed = holed | MorphClose{}.kernel_size(3);
    std::cout << std::format("MorphClose: hole pixel after close = {} (was 0)\n",
        (int)closed.mat().at<uchar>(10, 10));  // 255 — filled

    // ── Shape and iterations params ──────────────────────────────────────────
    Image<Gray> opened2 = noisy
        | MorphOpen{}.kernel_size(3).shape(MorphShape::Ellipse).iterations(1);
    std::cout << std::format("MorphOpen (Ellipse): {}x{}\n",
        opened2.cols(), opened2.rows());

    Image<Gray> closed2 = holed
        | MorphClose{}.kernel_size(3).shape(MorphShape::Cross).iterations(2);
    std::cout << std::format("MorphClose (Cross, 2 iters): {}x{}\n",
        closed2.cols(), closed2.rows());

    // ── BGR support ──────────────────────────────────────────────────────────
    cv::Mat bgr_mat(20, 20, CV_8UC3, cv::Scalar(0, 0, 0));
    bgr_mat.at<cv::Vec3b>(10, 10) = {255, 255, 255};
    Image<BGR> bgr_noisy{bgr_mat};
    Image<BGR> bgr_opened = bgr_noisy | MorphOpen{}.kernel_size(3);
    std::cout << std::format("MorphOpen on BGR: {}x{} BGR\n",
        bgr_opened.cols(), bgr_opened.rows());

    // ── Pipeline: open then close (morphological smoothing) ─────────────────
    Image<Gray> smoothed = noisy
        | MorphOpen{}.kernel_size(3)
        | MorphClose{}.kernel_size(3);
    std::cout << std::format("Open | Close pipeline: {}x{}\n",
        smoothed.cols(), smoothed.rows());

    std::cout << "Done.\n";
    return 0;
}
