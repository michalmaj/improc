/** @file img_hash.hpp
 *  @brief Perceptual image hash algorithms for similarity comparison.
 *
 *  All hash ops expose `operator()(Image<BGR>) → cv::Mat` and a static `distance()`.
 *  Hamming-based hashes return CV_8U; L2-based hashes return CV_64F.
 */
#pragma once
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

/**
 * @brief Average Hash — fast 64-bit perceptual hash based on mean pixel value.
 */
struct AverageHash {
    /// @return 1×8 CV_8U hash matrix (64 bits).
    cv::Mat operator()(const Image<BGR>&) const;
    /// @brief Computes the Hamming distance between two AverageHash digests.
    static double distance(const cv::Mat& a, const cv::Mat& b);
};

/**
 * @brief Perceptual Hash (pHash) — 64-bit DCT-based hash robust to minor image changes.
 */
struct PHash {
    /// @return 1×8 CV_8U hash matrix (64 bits, DCT-based).
    cv::Mat operator()(const Image<BGR>&) const;
    /// @brief Computes the Hamming distance between two PHash digests.
    static double distance(const cv::Mat& a, const cv::Mat& b);
};

/**
 * @brief Marr-Hildreth Hash — 576-bit hash based on LoG zero-crossings.
 */
struct MarrHildrethHash {
    /// @return 1×72 CV_8U hash matrix (576 bits, LoG zero-crossing).
    cv::Mat operator()(const Image<BGR>&) const;
    /// @brief Computes the Hamming distance between two MarrHildrethHash digests.
    static double distance(const cv::Mat& a, const cv::Mat& b);
};

/**
 * @brief Radial Variance Hash — L2-based hash using radial variance projections.
 */
struct RadialVarianceHash {
    /// @return 1×40 CV_64F hash matrix (radial variances).
    cv::Mat operator()(const Image<BGR>&) const;
    /// @brief Computes the L2 distance between two RadialVarianceHash digests.
    static double distance(const cv::Mat& a, const cv::Mat& b);
};

/**
 * @brief Color Moment Hash — L2-based hash encoding colour distribution moments.
 */
struct ColorMomentHash {
    /// @return 1×42 CV_64F hash matrix (color moments).
    cv::Mat operator()(const Image<BGR>&) const;
    /// @brief Computes the L2 distance between two ColorMomentHash digests.
    static double distance(const cv::Mat& a, const cv::Mat& b);
};

/**
 * @brief Block Mean Hash — 256-bit hash based on mean values of image blocks.
 */
struct BlockMeanHash {
    /// @return 1×32 CV_8U hash matrix (256 bits, block means).
    cv::Mat operator()(const Image<BGR>&) const;
    /// @brief Computes the Hamming distance between two BlockMeanHash digests.
    static double distance(const cv::Mat& a, const cv::Mat& b);
};

} // namespace improc::core
