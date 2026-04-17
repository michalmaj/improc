// include/improc/ml/dnn_forward.hpp
#pragma once

#include <string>
#include <unordered_set>
#include <vector>
#include <opencv2/dnn.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::ml {

using improc::core::Image;
using improc::core::BGR;

struct DnnForward {
    // Loads model at construction. Throws ModelError on failure.
    explicit DnnForward(std::string model_path);

    DnnForward& input_size(int w, int h) {
        if (w <= 0 || h <= 0) throw ParameterError{"input_size", "dimensions must be positive", "DnnForward"};
        input_w_ = w; input_h_ = h; return *this;
    }
    DnnForward& mean(cv::Scalar m)  { mean_ = m; return *this; }
    DnnForward& scale(float s) {
        if (s <= 0.0f) throw ParameterError{"scale", "must be positive", "DnnForward"};
        scale_ = s; return *this;
    }
    DnnForward& swap_rb(bool s)     { swap_rb_ = s; return *this; }

    // Runs forward pass. Returns flattened output tensor (all output values).
    std::vector<float> operator()(const Image<BGR>& img) const;

private:
    mutable cv::dnn::Net  net_;
    int                   input_w_ = 224;
    int                   input_h_ = 224;
    cv::Scalar            mean_    = {0, 0, 0};
    float                 scale_   = 1.0f / 255.0f;
    bool                  swap_rb_ = true;

    static const std::unordered_set<std::string>& valid_extensions();
};

} // namespace improc::ml
