// examples/core/demo_quality.cpp
//
// Demo: PSNR, SSIM, GMSD, MSE
//
// Compares an original synthetic image against: identical copy, Gaussian-
// blurred version, and a JPEG-compressed version. Prints a metric table.

#include "improc/core/pipeline.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <iomanip>

using namespace improc::core;

int main() {
    // ── Reference image ───────────────────────────────────────────────────────
    cv::Mat raw(300, 400, CV_8UC3, cv::Scalar(100, 120, 140));
    cv::rectangle(raw, {20,  20},  {180, 180}, {220, 80,  60}, -1);
    cv::circle(raw,    {300, 150}, 80,          {60,  180, 220}, -1);
    cv::Mat noise(raw.size(), CV_8UC3);
    cv::randu(noise, 0, 10);
    raw += noise;
    Image<BGR> ref(raw);

    // ── Variants ──────────────────────────────────────────────────────────────
    // Identical copy
    Image<BGR> identical(raw.clone());

    // Gaussian blur
    Image<BGR> blurred = ref | GaussianBlur{}
        .kernel_size(7)   // kernel side length, must be odd (default: 3)
        .sigma(3.0);      // Gaussian standard deviation (default: 0)

    // JPEG compression artefacts
    std::vector<uchar> buf;
    cv::imencode(".jpg", raw, buf, {cv::IMWRITE_JPEG_QUALITY, 20});
    cv::Mat jpeg_mat = cv::imdecode(buf, cv::IMREAD_COLOR);
    Image<BGR> jpeg(jpeg_mat);

    // ── Metrics ───────────────────────────────────────────────────────────────
    auto print_row = [&](const std::string& label, const Image<BGR>& cmp) {
        double psnr = PSNR{}(ref, cmp);
        double ssim = SSIM{}(ref, cmp);
        double gmsd = GMSD{}(ref, cmp);
        double mse  = MSE{} (ref, cmp);
        std::cout << std::left << std::setw(20) << label
                  << std::right
                  << std::setw(10) << std::fixed << std::setprecision(2) << psnr
                  << std::setw(8)  << std::setprecision(4) << ssim
                  << std::setw(10) << std::setprecision(4) << gmsd
                  << std::setw(10) << std::setprecision(2) << mse  << "\n";
    };

    std::cout << "\n"
              << std::left  << std::setw(20) << "Comparison"
              << std::right
              << std::setw(10) << "PSNR(dB)"
              << std::setw(8)  << "SSIM"
              << std::setw(10) << "GMSD"
              << std::setw(10) << "MSE" << "\n";
    std::cout << std::string(58, '-') << "\n";

    print_row("Identical",     identical);
    print_row("Blurred (σ=3)", blurred);
    print_row("JPEG Q=20",     jpeg);

    // ── Gray overloads ────────────────────────────────────────────────────────
    Image<Gray> ref_g  = ref | ToGray{};
    Image<Gray> blur_g = blurred | ToGray{};
    double psnr_gray = PSNR{}(ref_g, blur_g);
    double ssim_gray = SSIM{}(ref_g, blur_g);
    std::cout << "\nGray PSNR (blurred): " << psnr_gray << " dB\n";
    std::cout << "Gray SSIM (blurred): " << ssim_gray << "\n";

    std::cout << "\nDone.\n";
    return 0;
}
