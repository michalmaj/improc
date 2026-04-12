// include/improc/core/ops/normalize.hpp
#pragma once

#include <stdexcept>
#include "improc/core/image.hpp"

namespace improc::core {

struct Normalize {
    Image<Float32>   operator()(Image<Float32>   img) const;
    Image<Float32C3> operator()(Image<Float32C3> img) const;
};

struct NormalizeTo {
    explicit NormalizeTo(float min, float max);
    Image<Float32>   operator()(Image<Float32>   img) const;
    Image<Float32C3> operator()(Image<Float32C3> img) const;
private:
    float min_, max_;
};

struct Standardize {
    explicit Standardize(float mean, float std_dev);
    Image<Float32>   operator()(Image<Float32>   img) const;
    Image<Float32C3> operator()(Image<Float32C3> img) const;
private:
    float mean_, std_dev_;
};

} // namespace improc::core
