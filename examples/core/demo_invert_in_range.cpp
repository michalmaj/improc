// examples/core/demo_invert_in_range.cpp
// Demonstrates Invert and InRange — pixel-level binary operations.

#include <format>
#include <iostream>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

int main() {
    // ── Invert on Gray ───────────────────────────────────────────────────────
    cv::Mat gray_mat(100, 100, CV_8UC1, cv::Scalar(200));
    Image<Gray> gray{gray_mat};
    Image<Gray> inv_gray = gray | Invert{};
    std::cout << std::format("Gray 200 inverted → {}\n",
        (int)inv_gray.mat().at<uchar>(50, 50));   // 55

    // ── Invert on BGR ────────────────────────────────────────────────────────
    cv::Mat bgr_mat(100, 100, CV_8UC3, cv::Scalar(50, 100, 150));
    Image<BGR> bgr{bgr_mat};
    Image<BGR> inv_bgr = bgr | Invert{};
    cv::Vec3b px = inv_bgr.mat().at<cv::Vec3b>(50, 50);
    std::cout << std::format("BGR (50,100,150) inverted → ({},{},{})\n",
        (int)px[0], (int)px[1], (int)px[2]);      // (205,155,105)

    // ── Invert twice is identity ─────────────────────────────────────────────
    Image<Gray> restored = gray | Invert{} | Invert{};
    std::cout << std::format("Invert twice restores original: {}\n",
        (restored.mat().at<uchar>(50, 50) == 200) ? "yes" : "no");

    // ── InRange on Gray ──────────────────────────────────────────────────────
    // Pixels with value in [100, 200] → 255; outside → 0
    Image<Gray> gray_mask = gray | InRange{}.lower({100}).upper({200});
    std::cout << std::format("Gray InRange [100,200] on 200-image → {} white pixels\n",
        cv::countNonZero(gray_mask.mat()));        // 10000

    // ── InRange on BGR ───────────────────────────────────────────────────────
    // Select green-ish pixels (high G, low B and R)
    Image<Gray> bgr_mask = bgr | InRange{}.lower({0, 200, 0}).upper({50, 255, 50});
    std::cout << std::format("BGR InRange green-ish → {} white pixels\n",
        cv::countNonZero(bgr_mask.mat()));         // 0 (our BGR is (50,100,150))

    // ── Pipeline: InRange → ApplyMask ────────────────────────────────────────
    Image<Gray> all_mask = gray | InRange{}.lower({0}).upper({255});
    Image<Gray> selected = gray | ApplyMask{}.mask(all_mask);
    std::cout << std::format("InRange | ApplyMask: {}x{} Gray\n",
        selected.cols(), selected.rows());

    std::cout << "Done.\n";
    return 0;
}
