#pragma once
#include <vector>
#include <opencv2/stitching.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

struct StitchResult {
    bool       ok;
    Image<BGR> panorama;
};

struct Stitch {
    enum class Mode { Panorama, Scans };
    Stitch& mode(Mode m) { mode_ = m; return *this; }
    StitchResult operator()(const std::vector<Image<BGR>>&) const;
private:
    Mode mode_ = Mode::Panorama;
};

} // namespace improc::core
