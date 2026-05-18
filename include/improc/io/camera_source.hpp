// include/improc/io/camera_source.hpp
#pragma once

#include <expected>
#include "improc/io/camera_frame.hpp"
#include "improc/error.hpp"

namespace improc::io {

/**
 * @brief Concept satisfied by any type that can act as a camera source.
 *
 * A conforming type must provide:
 *   - `start()` — begin frame acquisition (e.g. open device, connect to stream).
 *   - `stop()`  — halt acquisition and release resources.
 *   - `getFrame()` — attempt to retrieve the next frame; returns `std::expected`
 *     so callers can handle Timeout / device errors without exceptions.
 *
 * All concrete camera sources (WebcamCapture, IPCameraCapture, OakDCapture)
 * must satisfy this concept.
 */
template<typename T>
concept CameraSourceType = requires(T t) {
    { t.start()    } -> std::same_as<void>;
    { t.stop()     } -> std::same_as<void>;
    { t.getFrame() } -> std::same_as<std::expected<CameraFrame, improc::Error>>;
};

} // namespace improc::io
