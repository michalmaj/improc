//
// Created by Michał Maj on 11/04/2026.
//

#include "improc/core/pipeline.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace improc::core;

int main() {
    // --- 1. Source image: a simple gradient BGR image ---
    cv::Mat raw(300, 400, CV_8UC3);
    for (int r = 0; r < raw.rows; ++r)
        for (int c = 0; c < raw.cols; ++c)
            raw.at<cv::Vec3b>(r, c) = {
                static_cast<uchar>(c * 255 / raw.cols),   // B
                static_cast<uchar>(r * 255 / raw.rows),   // G
                static_cast<uchar>(128)                    // R
            };

    Image<BGR> src(raw);
    std::cout << "Source:  Image<BGR>  " << src.cols() << "x" << src.rows() << "\n";
    cv::imshow("Step 0 — Source (BGR)", src.mat());
    cv::waitKey(0);

    // --- 2. Resize to 200x150 (both dims explicit) ---
    Image<BGR> resized = src | Resize{}.width(200).height(150);
    std::cout << "Resized: " << resized.cols() << "x" << resized.rows() << "\n";
    cv::imshow("Step 1 — Resize (200x150)", resized.mat());
    cv::waitKey(0);

    // --- 3. Resize preserving aspect ratio (width only) ---
    Image<BGR> aspect = src | Resize{}.width(100);
    std::cout << "Aspect ratio resize (width=100): " << aspect.cols() << "x" << aspect.rows() << "\n";

    // --- 4. Crop a region of interest ---
    Image<BGR> cropped = src | Crop{}.x(50).y(50).width(200).height(150);
    std::cout << "Cropped ROI: " << cropped.cols() << "x" << cropped.rows() << "\n";
    cv::imshow("Step 2 — Crop (200x150 at 50,50)", cropped.mat());
    cv::waitKey(0);

    // --- 5. Flip ---
    Image<BGR> flipped_h = src | Flip{Axis::Horizontal};
    Image<BGR> flipped_v = src | Flip{Axis::Vertical};
    cv::imshow("Step 3a — Flip Horizontal", flipped_h.mat());
    cv::imshow("Step 3b — Flip Vertical",   flipped_v.mat());
    cv::waitKey(0);

    // --- 6. Rotate 30 degrees around center ---
    Image<BGR> rotated = src | Rotate{}.angle(30.0);
    std::cout << "Rotated (30 deg): " << rotated.cols() << "x" << rotated.rows()
              << "  (output size preserved)\n";
    cv::imshow("Step 4 — Rotate 30°", rotated.mat());
    cv::waitKey(0);

    // --- 7. Full ML preprocessing pipeline ---
    //   BGR → Resize(224x224) → Flip(H) → Float32C3
    //   (normalization ops work on Image<Float32> single-channel only)
    Image<Float32C3> ml_ready =
        src
        | Resize{}.width(224).height(224)
        | Flip{Axis::Horizontal}
        | ToFloat32C3{};

    std::cout << "ML-ready: Image<Float32C3>  "
              << ml_ready.cols() << "x" << ml_ready.rows()
              << "  type=CV_32FC3  values in [0.0, 1.0]\n";
    cv::imshow("Step 5 — ML pipeline result (Float32C3, 0-1)", ml_ready.mat());
    cv::waitKey(0);

    // --- 8. Standardize (ImageNet-style, single channel demo) ---
    Image<Float32> gray_f =
        src | ToGray{} | ToFloat32{} | Normalize{};

    Image<Float32> standardized = gray_f | Standardize{0.485f, 0.229f};
    std::cout << "Standardized (mean=0.485, std=0.229): type=CV_32FC1\n";
    cv::imshow("Step 6 — Standardize (Gray Float32)", standardized.mat());
    cv::waitKey(0);

    return 0;
}
