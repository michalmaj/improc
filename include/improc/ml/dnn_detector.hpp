// include/improc/ml/dnn_detector.hpp
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

struct DnnDetector {
    enum class Style {
        YOLO,  // single output blob [1, N, 5+num_classes]: cx,cy,w,h,obj_conf,c0,c1,...
        SSD    // two output blobs: boxes [1,N,4] + scores [1,N,C] (y1,x1,y2,x2 normalized)
    };

    // Loads model at construction. Throws ModelError on failure.
    explicit DnnDetector(std::string model_path);

    DnnDetector& style(Style s)               { style_ = s; return *this; }
    DnnDetector& output_layer(std::string n)  { output_layer_ = std::move(n); return *this; }
    DnnDetector& boxes_layer(std::string n)   { boxes_layer_  = std::move(n); return *this; }
    DnnDetector& scores_layer(std::string n)  { scores_layer_ = std::move(n); return *this; }
    DnnDetector& confidence_threshold(float t) {
        if (t < 0.0f || t > 1.0f) throw ParameterError{"confidence_threshold", "must be in [0, 1]", "DnnDetector"};
        confidence_threshold_ = t; return *this;
    }
    DnnDetector& nms_threshold(float t) {
        if (t < 0.0f || t > 1.0f) throw ParameterError{"nms_threshold", "must be in [0, 1]", "DnnDetector"};
        nms_threshold_ = t; return *this;
    }
    DnnDetector& input_size(int w, int h) {
        if (w <= 0 || h <= 0) throw ParameterError{"input_size", "dimensions must be positive", "DnnDetector"};
        input_w_ = w; input_h_ = h; return *this;
    }
    DnnDetector& mean(cv::Scalar m)                     { mean_ = m; return *this; }
    DnnDetector& scale(float s) {
        if (s <= 0.0f) throw ParameterError{"scale", "must be positive", "DnnDetector"};
        scale_ = s; return *this;
    }
    DnnDetector& swap_rb(bool s)                        { swap_rb_ = s; return *this; }
    DnnDetector& labels(std::vector<std::string> l)     { labels_ = std::move(l); return *this; }

    // Throws Exception if inference fails.
    std::vector<Detection> operator()(const Image<BGR>& img) const;

private:
    mutable cv::dnn::Net     net_;
    Style                    style_                = Style::YOLO;
    std::string              output_layer_;
    std::string              boxes_layer_;
    std::string              scores_layer_;
    float                    confidence_threshold_ = 0.5f;
    float                    nms_threshold_        = 0.4f;
    int                      input_w_             = 640;
    int                      input_h_             = 640;
    cv::Scalar               mean_                = {0, 0, 0};
    float                    scale_               = 1.0f / 255.0f;
    bool                     swap_rb_             = true;
    std::vector<std::string> labels_;

    std::vector<Detection> parse_yolo(const cv::Mat& output, int orig_w, int orig_h) const;
    std::vector<Detection> parse_ssd(const cv::Mat& boxes, const cv::Mat& scores, int orig_w, int orig_h) const;

    static const std::unordered_set<std::string>& valid_extensions();
};

} // namespace improc::ml
