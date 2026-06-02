// include/improc/core/ops/draw_matches.hpp
// DrawKeypoints and DrawMatches have moved to improc::visualization.
// This header is kept for backward compatibility.
#pragma once
#include "improc/visualization/draw_matches.hpp"
namespace improc::core {
    // Deprecated: use improc::visualization::DrawKeypoints instead.
    using improc::visualization::DrawKeypoints;
    // Deprecated: use improc::visualization::DrawMatches instead.
    using improc::visualization::DrawMatches;
} // namespace improc::core
