// src/core/ops/to_lab.cpp
#include "improc/core/ops/to_lab.hpp"
#include <utility>

namespace improc::core {

Image<LAB> ToLAB::operator()(Image<BGR> img) const {
    return convert<LAB, BGR>(std::move(img));
}

} // namespace improc::core
