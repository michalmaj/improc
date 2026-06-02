#pragma once
#include <opencv2/core.hpp>

namespace improc::core {

/**
 * @brief Typed wrapper around a perceptual hash matrix.
 *
 * All improc hash ops return this type. Use the static distance() method
 * on the originating hash struct to compare two hashes.
 */
struct ImageHash {
    cv::Mat value;

    bool empty() const { return value.empty(); }

    bool operator==(const ImageHash& o) const {
        if (value.size() != o.value.size() || value.type() != o.value.type()) return false;
        return cv::norm(value, o.value, cv::NORM_L1) == 0.0;
    }
    bool operator!=(const ImageHash& o) const { return !(*this == o); }
};

} // namespace improc::core
