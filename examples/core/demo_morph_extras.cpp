// examples/core/demo_morph_extras.cpp
// Demonstrates MorphGradient (boundary highlight), TopHat (small bright features),
// and BlackHat (small dark features).

#include <format>
#include <iostream>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

int main() {
    // ── MorphGradient: highlights object boundaries (dilate − erode) ─────────
    cv::Mat rect_mat(30, 30, CV_8UC1, cv::Scalar(0));
    rect_mat(cv::Rect(5, 5, 20, 20)) = 255;   // white rectangle on black
    Image<Gray> rect{rect_mat};

    Image<Gray> grad = rect | MorphGradient{}.kernel_size(3);
    // Interior of rectangle is 0; boundary pixels are bright
    int interior = grad.mat().at<uchar>(15, 15);
    int boundary  = grad.mat().at<uchar>(5,  5);
    std::cout << std::format("MorphGradient: interior={} boundary={} (boundary > interior: {})\n",
        interior, boundary, boundary > interior ? "true" : "false");

    // Uniform image → zero gradient everywhere
    Image<Gray> flat{cv::Mat(20, 20, CV_8UC1, cv::Scalar(128))};
    Image<Gray> grad_flat = flat | MorphGradient{};
    std::cout << std::format("MorphGradient on flat image: center={} (expected 0)\n",
        (int)grad_flat.mat().at<uchar>(10, 10));

    // BGR support
    Image<BGR> bgr_rect{cv::Mat(30, 30, CV_8UC3, cv::Scalar(50, 100, 150))};
    Image<BGR> bgr_grad = bgr_rect | MorphGradient{}.kernel_size(3);
    std::cout << std::format("MorphGradient on BGR: {}x{} BGR\n",
        bgr_grad.cols(), bgr_grad.rows());

    // ── TopHat: reveals bright features smaller than the kernel ──────────────
    // Large bright background with a small bright spot — TopHat isolates the spot
    cv::Mat base_mat(30, 30, CV_8UC1, cv::Scalar(100));
    base_mat.at<uchar>(15, 15) = 200;   // single bright pixel above background
    Image<Gray> base{base_mat};

    Image<Gray> tophat = base | TopHat{}.kernel_size(3);
    int spot    = tophat.mat().at<uchar>(15, 15);
    int bg      = tophat.mat().at<uchar>(2,  2);
    std::cout << std::format("TopHat: bright spot={} background={} (spot > bg: {})\n",
        spot, bg, spot > bg ? "true" : "false");

    // Uniform image → zero everywhere
    Image<Gray> th_flat = flat | TopHat{};
    std::cout << std::format("TopHat on flat image: center={} (expected 0)\n",
        (int)th_flat.mat().at<uchar>(10, 10));

    // ── BlackHat: reveals dark features smaller than the kernel ──────────────
    // Bright background with a single dark hole — BlackHat isolates the hole
    cv::Mat bright_mat(30, 30, CV_8UC1, cv::Scalar(200));
    bright_mat.at<uchar>(15, 15) = 50;   // single dark pixel below background
    Image<Gray> bright{bright_mat};

    Image<Gray> blackhat = bright | BlackHat{}.kernel_size(3);
    int hole   = blackhat.mat().at<uchar>(15, 15);
    int surr   = blackhat.mat().at<uchar>(2,  2);
    std::cout << std::format("BlackHat: dark hole={} surrounding={} (hole > surrounding: {})\n",
        hole, surr, hole > surr ? "true" : "false");

    // Uniform image → zero everywhere
    Image<Gray> bh_flat = flat | BlackHat{};
    std::cout << std::format("BlackHat on flat image: center={} (expected 0)\n",
        (int)bh_flat.mat().at<uchar>(10, 10));

    // ── Shape parameter ───────────────────────────────────────────────────────
    Image<Gray> grad_cross = rect | MorphGradient{}.kernel_size(3).shape(MorphShape::Cross);
    Image<Gray> grad_ellip = rect | MorphGradient{}.kernel_size(3).shape(MorphShape::Ellipse);
    std::cout << std::format("MorphGradient Cross: {}x{}  Ellipse: {}x{}\n",
        grad_cross.cols(), grad_cross.rows(),
        grad_ellip.cols(), grad_ellip.rows());

    // ── Pipeline: gradient → threshold to get clean boundary mask ────────────
    Image<Gray> boundary_mask = rect
        | MorphGradient{}.kernel_size(3)
        | Threshold{}.value(1).mode(ThresholdMode::Binary);
    int mask_boundary = boundary_mask.mat().at<uchar>(5, 5);
    int mask_interior = boundary_mask.mat().at<uchar>(15, 15);
    std::cout << std::format("MorphGradient | Threshold: boundary={} interior={}\n",
        mask_boundary, mask_interior);

    std::cout << "Done.\n";
    return 0;
}
