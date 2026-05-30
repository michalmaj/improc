// include/improc/io/camera_capture.hpp
#pragma once
/** @file camera_capture.hpp
 *  @brief Backward-compatibility alias: `improc::io::CameraCapture` → `WebcamCapture`.
 *  @deprecated New code should use `WebcamCapture` directly.
 */
#include "improc/io/webcam_capture.hpp"

namespace improc::io {
using CameraCapture = WebcamCapture;
}
