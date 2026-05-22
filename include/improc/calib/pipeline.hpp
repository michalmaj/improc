// include/improc/calib/pipeline.hpp
#pragma once

// operator| and Image<F> live in improc::core; include them so callers get
// both namespaces from a single include and ADL finds operator| correctly.
#include "improc/core/pipeline.hpp"
#include "improc/calib/ops/calib_types.hpp"
