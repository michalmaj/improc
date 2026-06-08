// include/improc/onnx/onnx.hpp
#pragma once

#ifndef IMPROC_WITH_ONNX
#  error "improc/onnx/ requires ONNX Runtime — configure with -DIMPROC_WITH_ONNX=ON"
#endif

/** @file onnx.hpp
 *  @brief Convenience header — includes all `improc::onnx` types:
 *         OnnxSession, OnnxClassifier, OnnxDetector, OnnxSegmentor, OnnxInstanceSegmentor.
 *
 *  Individual headers are fully documented; this umbrella exists for single-include convenience.
 *  Requires building with @c IMPROC_WITH_ONNX=ON (the default).
 */

#include "improc/onnx/onnx_session.hpp"
#include "improc/onnx/onnx_classifier.hpp"
#include "improc/onnx/onnx_detector.hpp"
#include "improc/onnx/onnx_segmentor.hpp"
#include "improc/onnx/onnx_instance_segmentor.hpp"
