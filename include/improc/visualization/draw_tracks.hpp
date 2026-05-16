// include/improc/visualization/draw_tracks.hpp
#pragma once

#include <format>
#include <vector>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/ml/tracking/track.hpp"
#include "improc/exceptions.hpp"

namespace improc::visualization {

using improc::core::Image;
using improc::core::BGR;

/**
 * @brief Draws `Track` bounding boxes with ID labels onto an `Image<BGR>`.
 *
 * Returns a new annotated image; the source is not modified.
 *
 * @code
 * Image<BGR> annotated = frame | DrawTracks{tracks}.thickness(2);
 * @endcode
 */
struct DrawTracks {
    explicit DrawTracks(std::vector<improc::ml::Track> tracks)
        : tracks_(std::move(tracks)) {}

    /// @brief Sets the box and label color as a BGR scalar.
    DrawTracks& color(cv::Scalar c)  { color_ = c; return *this; }
    /// @brief Sets the rectangle and text line thickness.
    /// @throws improc::ParameterError if `t` <= 0.
    DrawTracks& thickness(int t) {
        if (t <= 0) throw ParameterError{"thickness", "must be positive", "DrawTracks"};
        thickness_ = t;
        return *this;
    }
    /// @brief Sets the font scale for the ID label.
    /// @throws improc::ParameterError if `s` <= 0.
    DrawTracks& font_scale(double s) {
        if (s <= 0.0) throw ParameterError{"font_scale", "must be positive", "DrawTracks"};
        font_scale_ = s;
        return *this;
    }
    /// @brief Enables or disables drawing the track ID label (default: true).
    DrawTracks& show_id(bool b) { show_id_ = b; return *this; }

    /// @brief Draws all tracks onto a clone of `img` and returns the annotated copy.
    Image<BGR> operator()(Image<BGR> img) const {
        cv::Mat canvas = img.mat().clone();

        for (const auto& t : tracks_) {
            cv::Rect box(
                static_cast<int>(t.bbox.box.x),
                static_cast<int>(t.bbox.box.y),
                static_cast<int>(t.bbox.box.width),
                static_cast<int>(t.bbox.box.height));

            cv::rectangle(canvas, box, color_, thickness_);

            if (show_id_) {
                std::string text = std::format("ID:{}", t.id);
                const cv::Point origin{box.x, std::max(box.y - 4, 0)};
                cv::putText(canvas, text, origin,
                            cv::FONT_HERSHEY_SIMPLEX, font_scale_,
                            color_, thickness_);
            }
        }

        return Image<BGR>(std::move(canvas));
    }

private:
    std::vector<improc::ml::Track> tracks_;
    cv::Scalar color_      = {255, 255, 0};  // cyan
    int        thickness_  = 2;
    double     font_scale_ = 0.5;
    bool       show_id_    = true;
};

} // namespace improc::visualization
