// include/improc/improc.hpp
#pragma once

/** @file improc.hpp
 *  @brief Top-level convenience header — includes all `improc` namespaces.
 *
 *  Pull in everything with a single include:
 *  @code
 *  #include "improc/improc.hpp"
 *  @endcode
 *
 *  For production builds prefer the individual namespace headers
 *  (`improc/core/pipeline.hpp`, `improc/ml/ml.hpp`, …) to keep
 *  compile times low.
 */

#include "improc/version.hpp"
#include "improc/error.hpp"
#include "improc/exceptions.hpp"

#include "improc/core/pipeline.hpp"
#include "improc/views/views.hpp"
#include "improc/io/io.hpp"
#include "improc/calib/calib.hpp"
#include "improc/ml/ml.hpp"
#include "improc/onnx/onnx.hpp"
#include "improc/threading/threading.hpp"
#include "improc/visualization/visualization.hpp"
