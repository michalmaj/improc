// include/improc/core/ops/axis.hpp
#pragma once

namespace improc::core {

/// @brief Axis selector for Flip and similar ops.
enum class Axis {
    Horizontal, ///< Flip left-right (around the vertical axis).
    Vertical,   ///< Flip top-bottom (around the horizontal axis).
    Both        ///< Flip both axes simultaneously (equivalent to 180° rotation).
};

} // namespace improc::core
