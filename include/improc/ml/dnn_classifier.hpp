// include/improc/ml/dnn_classifier.hpp
#pragma once

#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>
#include <opencv2/dnn.hpp>
#include "improc/core/image.hpp"
#include "improc/ml/result_types.hpp"
#include "improc/exceptions.hpp"

namespace improc::ml {

using improc::core::Image;
using improc::core::BGR;

struct DnnClassifier {
    // Loads model at construction. Throws ModelError if file does not exist,
    // has an unsupported extension, or OpenCV fails to parse it.
    explicit DnnClassifier(std::string model_path);

    DnnClassifier& top_k(int k) {
        if (k <= 0) throw ParameterError{"top_k", "must be positive", "DnnClassifier"};
        top_k_ = k; return *this;
    }
    DnnClassifier& input_size(int w, int h) {
        if (w <= 0 || h <= 0) throw ParameterError{"input_size", "dimensions must be positive", "DnnClassifier"};
        input_w_ = w; input_h_ = h; return *this;
    }
    DnnClassifier& mean(cv::Scalar m)   { mean_ = m; return *this; }
    DnnClassifier& scale(float s) {
        if (s <= 0.0f) throw ParameterError{"scale", "must be positive", "DnnClassifier"};
        scale_ = s; return *this;
    }
    DnnClassifier& swap_rb(bool s)                      { swap_rb_ = s; return *this; }
    DnnClassifier& labels(std::vector<std::string> l)   { labels_ = std::move(l); return *this; }

    std::vector<ClassResult> operator()(const Image<BGR>& img) const;

private:
    mutable cv::dnn::Net     net_;
    int                      top_k_   = 5;
    int                      input_w_ = 224;
    int                      input_h_ = 224;
    cv::Scalar               mean_    = {0, 0, 0};
    float                    scale_   = 1.0f / 255.0f;
    bool                     swap_rb_ = true;
    std::vector<std::string> labels_;

    static const std::unordered_set<std::string>& valid_extensions();
};

} // namespace improc::ml
