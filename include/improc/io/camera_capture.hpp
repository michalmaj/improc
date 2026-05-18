// include/improc/io/camera_capture.hpp
// Backward-compat alias — existing code using CameraCapture compiles unchanged.
// New code should use WebcamCapture directly.
#pragma once
#include "improc/io/webcam_capture.hpp"

namespace improc::io {
using CameraCapture = WebcamCapture;
}
