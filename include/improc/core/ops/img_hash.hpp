#pragma once
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

// All hash ops: operator()(Image<BGR>) → cv::Mat hash.
// Static distance() uses the metric native to each algorithm.
// Hamming-based hashes return CV_8U; L2-based return CV_64F.

struct AverageHash {
    cv::Mat operator()(const Image<BGR>&) const; // → 1×8 CV_8U (64 bits)
    static double distance(const cv::Mat& a, const cv::Mat& b); // Hamming
};

struct PHash {
    cv::Mat operator()(const Image<BGR>&) const; // → 1×8 CV_8U (64 bits, DCT-based)
    static double distance(const cv::Mat& a, const cv::Mat& b); // Hamming
};

struct MarrHildrethHash {
    cv::Mat operator()(const Image<BGR>&) const; // → 1×72 CV_8U (LoG zero-crossing)
    static double distance(const cv::Mat& a, const cv::Mat& b); // Hamming
};

struct RadialVarianceHash {
    cv::Mat operator()(const Image<BGR>&) const; // → 1×40 CV_64F (radial variances)
    static double distance(const cv::Mat& a, const cv::Mat& b); // L2
};

struct ColorMomentHash {
    cv::Mat operator()(const Image<BGR>&) const; // → 1×42 CV_64F (color moments)
    static double distance(const cv::Mat& a, const cv::Mat& b); // L2
};

struct BlockMeanHash {
    cv::Mat operator()(const Image<BGR>&) const; // → 1×32 CV_8U (256 bits, block means)
    static double distance(const cv::Mat& a, const cv::Mat& b); // Hamming
};

} // namespace improc::core
