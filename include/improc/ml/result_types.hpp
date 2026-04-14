// include/improc/ml/result_types.hpp
#pragma once

#include <string>
#include <opencv2/core.hpp>

namespace improc::ml {

struct ClassResult {
    int         class_id;  // index into labels vector (or raw output index)
    float       score;     // softmax probability or raw logit
    std::string label;     // empty if no labels were provided
};

struct Detection {
    cv::Rect2f  box;        // pixel coordinates on the original input image
    int         class_id;
    float       confidence;
    std::string label;      // empty if no labels were provided
};

} // namespace improc::ml
