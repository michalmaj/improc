/**
 * @brief Umbrella include for all `improc::calib` pipeline ops.
 *
 * Including this header pulls in the calib result types (FindChessboardResult,
 * CalibrationResult, UndistortMapResult, PnPResult, PnPRansacResult) and the
 * core pipeline infrastructure so that callers get both namespaces from a
 * single include and ADL finds `operator|` correctly.
 */
// include/improc/calib/pipeline.hpp
#pragma once

// operator| and Image<F> live in improc::core; include them so callers get
// both namespaces from a single include and ADL finds operator| correctly.
#include "improc/core/pipeline.hpp"
#include "improc/calib/ops/calib_types.hpp"
#include "improc/calib/ops/chessboard.hpp"
#include "improc/calib/ops/calibrate.hpp"
#include "improc/calib/ops/undistort.hpp"
#include "improc/calib/ops/project.hpp"
