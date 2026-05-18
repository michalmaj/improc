// include/improc/io/camera_frame.hpp
#pragma once

#include <chrono>
#include <optional>
#include <string>
#include "improc/core/image.hpp"

namespace improc::io {

/**
 * @brief Snapshot from a camera source, optionally with aligned metric depth.
 *
 * Produced by all camera source types in the real-time pipeline.
 * `rgb` is always present; `depth` is populated only by depth-capable devices
 * (e.g. OAK-D, RealSense). `timestamp` records capture time using the
 * steady clock to avoid wall-clock jumps. `source_id` is a human-readable
 * identifier for the originating device (e.g. "webcam:0", "oak-d",
 * "rtsp://...").
 */
struct CameraFrame {
    std::optional<core::Image<core::BGR>>                rgb;        ///< Colour frame (always populated once captured).
    std::optional<core::Image<core::Float32>>            depth;      ///< Metric depth in metres; nullopt = no depth sensor.
    std::chrono::steady_clock::time_point                timestamp;  ///< Capture time (steady clock).
    std::string                                          source_id;  ///< E.g. "webcam:0", "oak-d", "rtsp://...".
};

} // namespace improc::io
