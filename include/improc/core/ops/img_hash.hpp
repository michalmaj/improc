/** @file img_hash.hpp
 *  @brief Perceptual image hash algorithms for similarity comparison.
 *
 *  All hash ops expose `operator()(Image<BGR>) → ImageHash` and a static `distance()`.
 *  Hamming-based hashes store CV_8U internally; L2-based hashes store CV_64F internally.
 */
#pragma once
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/types/image_hash.hpp"

namespace improc::core {

/**
 * @brief Average Hash — fast 64-bit perceptual hash based on mean pixel value.
 */
struct AverageHash {
    /// @return ImageHash wrapping a 1×8 CV_8U hash matrix (64 bits).
    [[nodiscard]] ImageHash operator()(const Image<BGR>&) const;
    /// @brief Computes the Hamming distance between two AverageHash digests.
    static double distance(const ImageHash& a, const ImageHash& b);
};

/**
 * @brief Perceptual Hash (pHash) — 64-bit DCT-based hash robust to minor image changes.
 */
struct PHash {
    /// @return ImageHash wrapping a 1×8 CV_8U hash matrix (64 bits, DCT-based).
    [[nodiscard]] ImageHash operator()(const Image<BGR>&) const;
    /// @brief Computes the Hamming distance between two PHash digests.
    static double distance(const ImageHash& a, const ImageHash& b);
};

/**
 * @brief Marr-Hildreth Hash — 576-bit hash based on LoG zero-crossings.
 */
struct MarrHildrethHash {
    /// @return ImageHash wrapping a 1×72 CV_8U hash matrix (576 bits, LoG zero-crossing).
    [[nodiscard]] ImageHash operator()(const Image<BGR>&) const;
    /// @brief Computes the Hamming distance between two MarrHildrethHash digests.
    static double distance(const ImageHash& a, const ImageHash& b);
};

/**
 * @brief Radial Variance Hash — L2-based hash using radial variance projections.
 */
struct RadialVarianceHash {
    /// @return ImageHash wrapping a 1×40 CV_64F hash matrix (radial variances).
    [[nodiscard]] ImageHash operator()(const Image<BGR>&) const;
    /// @brief Computes the L2 distance between two RadialVarianceHash digests.
    static double distance(const ImageHash& a, const ImageHash& b);
};

/**
 * @brief Color Moment Hash — L2-based hash encoding colour distribution moments.
 */
struct ColorMomentHash {
    /// @return ImageHash wrapping a 1×42 CV_64F hash matrix (color moments).
    [[nodiscard]] ImageHash operator()(const Image<BGR>&) const;
    /// @brief Computes the L2 distance between two ColorMomentHash digests.
    static double distance(const ImageHash& a, const ImageHash& b);
};

/**
 * @brief Block Mean Hash — 256-bit hash based on mean values of image blocks.
 */
struct BlockMeanHash {
    /// @return ImageHash wrapping a 1×32 CV_8U hash matrix (256 bits, block means).
    [[nodiscard]] ImageHash operator()(const Image<BGR>&) const;
    /// @brief Computes the Hamming distance between two BlockMeanHash digests.
    static double distance(const ImageHash& a, const ImageHash& b);
};

} // namespace improc::core
