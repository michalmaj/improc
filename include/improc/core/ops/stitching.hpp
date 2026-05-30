#pragma once
#include <vector>
#include <opencv2/stitching.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Result of image stitching.
 */
struct StitchResult {
    bool       ok;       ///< True if stitching succeeded; false if feature matching or blending failed.
    Image<BGR> panorama; ///< The stitched panorama image (valid only when ok is true).
};

/**
 * @brief Stitches a sequence of overlapping BGR images into a single panorama.
 *
 * @code
 * auto result = Stitch().mode(Stitch::Mode::Panorama)(images);
 * if (result.ok) display(result.panorama);
 * @endcode
 */
struct Stitch {
    /// @brief Stitching mode.
    enum class Mode {
        Panorama, ///< Full 360° panorama mode using spherical warp (default).
        Scans,    ///< Flat document / scan stitching mode using affine warp.
    };
    /// @brief Sets the stitching mode (default: Mode::Panorama).
    Stitch& mode(Mode m) { mode_ = m; return *this; }
    /// @return StitchResult; ok is false if feature matching or blending fails.
    StitchResult operator()(const std::vector<Image<BGR>>&) const;
private:
    Mode mode_ = Mode::Panorama;
};

} // namespace improc::core
