// include/improc/visualization/draw.hpp
#pragma once

#include <format>
#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/ml/result_types.hpp"
#include "improc/exceptions.hpp"

namespace improc::visualization {

using improc::core::Image;
using improc::core::BGR;

// Draws bounding boxes (and optional labels / confidence scores) onto a copy
// of the source image. Pipeline-compatible: returns the annotated Image<BGR>.
//
// Usage:
//   auto detections = detector(frame);
//   Image<BGR> annotated = frame | DrawBoundingBoxes{detections}.thickness(2);
struct DrawBoundingBoxes {
    explicit DrawBoundingBoxes(std::vector<improc::ml::Detection> detections)
        : detections_(std::move(detections)) {}

    DrawBoundingBoxes& color(cv::Scalar c)     { color_ = c; return *this; }
    DrawBoundingBoxes& thickness(int t) {
        if (t <= 0) throw ParameterError{"thickness", "must be positive", "DrawBoundingBoxes"};
        thickness_ = t;
        return *this;
    }
    DrawBoundingBoxes& font_scale(double s) {
        if (s <= 0.0) throw ParameterError{"font_scale", "must be positive", "DrawBoundingBoxes"};
        font_scale_ = s;
        return *this;
    }
    DrawBoundingBoxes& show_label(bool b)      { show_label_ = b;      return *this; }
    DrawBoundingBoxes& show_confidence(bool b) { show_confidence_ = b; return *this; }

    // Draws onto a clone of img and returns it.
    Image<BGR> operator()(Image<BGR> img) const {
        cv::Mat canvas = img.mat().clone();

        for (const auto& det : detections_) {
            cv::Rect box(
                static_cast<int>(det.box.x),
                static_cast<int>(det.box.y),
                static_cast<int>(det.box.width),
                static_cast<int>(det.box.height));

            cv::rectangle(canvas, box, color_, thickness_);

            if (show_label_ || show_confidence_) {
                std::string text;
                if (show_label_ && !det.label.empty())
                    text = det.label;
                else if (show_label_)
                    text = std::format("id:{}", det.class_id);
                if (show_confidence_)
                    text += std::format("{}{:.2f}", text.empty() ? "" : " ", det.confidence);

                if (!text.empty()) {
                    const cv::Point origin{box.x, std::max(box.y - 4, 0)};
                    cv::putText(canvas, text, origin,
                                cv::FONT_HERSHEY_SIMPLEX, font_scale_,
                                color_, thickness_);
                }
            }
        }

        return Image<BGR>(std::move(canvas));
    }

private:
    std::vector<improc::ml::Detection> detections_;
    cv::Scalar color_          = {0, 255, 0};  // green
    int        thickness_      = 2;
    double     font_scale_     = 0.5;
    bool       show_label_     = true;
    bool       show_confidence_ = true;
};

} // namespace improc::visualization
