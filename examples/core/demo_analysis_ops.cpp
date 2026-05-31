// examples/core/demo_analysis_ops.cpp
//
// Demo: IntegralImage, MinMaxLoc, MeanStdDev, CountNonZero, Reduce, Moments

#include <format>
#include <iostream>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

int main() {
    cv::Mat m(50, 50, CV_8UC1, cv::Scalar(128));
    m.at<uchar>(10, 10) = 0;    // known minimum
    m.at<uchar>(40, 40) = 255;  // known maximum
    Image<Gray> img(m);

    // ── IntegralImage ─────────────────────────────────────────────────────────
    auto integral = IntegralImage{}(img);
    std::cout << std::format("IntegralImage sum at (50,50): {}\n",
                             integral.sum.at<int>(50, 50));

    auto integral_sq = IntegralImage{}.with_sq_sum(true)(img);
    std::cout << std::format("Squared integral empty? {}\n",
                             integral_sq.sq_sum.empty() ? "yes" : "no");

    // ── MinMaxLoc ─────────────────────────────────────────────────────────────
    auto mmr = MinMaxLoc{}(img);
    std::cout << std::format("MinMaxLoc min={} at ({},{})\n",
                             mmr.min_val, mmr.min_loc.x, mmr.min_loc.y);
    std::cout << std::format("MinMaxLoc max={} at ({},{})\n",
                             mmr.max_val, mmr.max_loc.x, mmr.max_loc.y);

    // ── MeanStdDev ────────────────────────────────────────────────────────────
    auto msd = MeanStdDev{}(img);
    std::cout << std::format("MeanStdDev mean≈{:.1f} stddev≈{:.2f}\n",
                             msd.mean[0], msd.stddev[0]);

    // ── CountNonZero ──────────────────────────────────────────────────────────
    cv::Mat bin(20, 20, CV_8UC1, cv::Scalar(0));
    bin.at<uchar>(5, 5) = 255;
    bin.at<uchar>(10, 10) = 255;
    Image<Gray> bin_img(bin);
    std::cout << std::format("CountNonZero: {} (expected 2)\n",
                             CountNonZero{}(bin_img));

    // ── Reduce ────────────────────────────────────────────────────────────────
    cv::Mat rmat(5, 5, CV_8UC1, cv::Scalar(4));
    Image<Gray> rimg(rmat);
    cv::Mat row_sum = Reduce{}.op(ReduceOp::Sum).dim(0)(rimg);
    std::cout << std::format("Reduce sum of cols (5x4): {} (expected 20)\n",
                             row_sum.at<int>(0, 0));

    // ── Moments ───────────────────────────────────────────────────────────────
    cv::Mat sq(100, 100, CV_8UC1, cv::Scalar(0));
    cv::rectangle(sq, {20, 20}, {80, 80}, cv::Scalar(255), -1);
    Image<Gray> sq_img(sq);
    auto mom = Moments{}(sq_img);
    double cx = mom.m10 / mom.m00;
    double cy = mom.m01 / mom.m00;
    std::cout << std::format("Moments centroid: ({:.1f}, {:.1f}) expected (50, 50)\n",
                             cx, cy);
    double hu[7];
    cv::HuMoments(mom, hu);
    std::cout << std::format("Hu moment[0]: {:.6f}\n", hu[0]);

    std::cout << "demo_analysis_ops: OK\n";
    return 0;
}
