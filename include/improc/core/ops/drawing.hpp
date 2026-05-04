// include/improc/core/ops/drawing.hpp
#pragma once
#include <string>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Pipeline op: renders text onto a BGR image.
 *
 * Uses `cv::putText` with `FONT_HERSHEY_SIMPLEX` and `LINE_AA`.
 * The source image is never mutated — a clone is drawn on.
 *
 * @throws improc::ParameterError if `font_scale <= 0` or `thickness <= 0`.
 *
 * @code
 * Image<BGR> annotated = img | DrawText{"Score: 0.95"}.position({10, 30}).color({0, 255, 0});
 * @endcode
 */
struct DrawText {
    explicit DrawText(std::string text) : text_(std::move(text)) {}

    DrawText& position(cv::Point p) { position_ = p; return *this; }

    DrawText& font_scale(double s) {
        if (s <= 0.0)
            throw ParameterError{"font_scale", "must be positive", "DrawText"};
        font_scale_ = s;
        return *this;
    }

    DrawText& color(cv::Scalar c) { color_ = c; return *this; }

    DrawText& thickness(int t) {
        if (t <= 0)
            throw ParameterError{"thickness", "must be positive", "DrawText"};
        thickness_ = t;
        return *this;
    }

    Image<BGR> operator()(Image<BGR> img) const;

private:
    std::string text_;
    cv::Point   position_{10, 30};
    double      font_scale_{1.0};
    cv::Scalar  color_{0, 255, 0};
    int         thickness_{1};
};

/**
 * @brief Pipeline op: draws a line onto a BGR image.
 *
 * Uses `cv::line` with `LINE_AA`. The source image is never mutated.
 *
 * @throws improc::ParameterError if `thickness <= 0`.
 *
 * @code
 * Image<BGR> result = img | DrawLine{{0, 0}, {100, 100}}.color({255, 0, 0}).thickness(2);
 * @endcode
 */
struct DrawLine {
    DrawLine(cv::Point p1, cv::Point p2) : p1_(p1), p2_(p2) {}

    DrawLine& color(cv::Scalar c) { color_ = c; return *this; }

    DrawLine& thickness(int t) {
        if (t <= 0)
            throw ParameterError{"thickness", "must be positive", "DrawLine"};
        thickness_ = t;
        return *this;
    }

    Image<BGR> operator()(Image<BGR> img) const;

private:
    cv::Point  p1_, p2_;
    cv::Scalar color_{0, 255, 0};
    int        thickness_{1};
};

/**
 * @brief Pipeline op: draws a circle onto a BGR image.
 *
 * Uses `cv::circle` with `LINE_AA`. The source image is never mutated.
 * Pass `thickness(-1)` to fill the circle.
 *
 * @throws improc::ParameterError if `radius <= 0` (at construction) or
 *         if `thickness` is not positive and not -1.
 *
 * @code
 * Image<BGR> result = img | DrawCircle{{150, 100}, 50}.color({0, 0, 255});
 * Image<BGR> filled = img | DrawCircle{{150, 100}, 50}.thickness(-1);  // filled
 * @endcode
 */
struct DrawCircle {
    DrawCircle(cv::Point center, int radius) : center_(center), radius_(radius) {
        if (radius <= 0)
            throw ParameterError{"radius", "must be positive", "DrawCircle"};
    }

    DrawCircle& color(cv::Scalar c) { color_ = c; return *this; }

    DrawCircle& thickness(int t) {
        if (t <= 0 && t != -1)
            throw ParameterError{"thickness", "must be positive or -1 (fill)", "DrawCircle"};
        thickness_ = t;
        return *this;
    }

    Image<BGR> operator()(Image<BGR> img) const;

private:
    cv::Point  center_;
    int        radius_;
    cv::Scalar color_{0, 255, 0};
    int        thickness_{1};
};

/**
 * @brief Pipeline op: draws a rectangle onto a BGR image.
 *
 * Uses `cv::rectangle` with `LINE_AA`. The source image is never mutated.
 * Pass `thickness(-1)` to fill the rectangle.
 *
 * @throws improc::ParameterError if `thickness` is not positive and not -1.
 *
 * @code
 * Image<BGR> result = img | DrawRectangle{cv::Rect{10, 10, 100, 80}}.color({255, 255, 0});
 * @endcode
 */
struct DrawRectangle {
    explicit DrawRectangle(cv::Rect r) : rect_(r) {}

    DrawRectangle& color(cv::Scalar c) { color_ = c; return *this; }

    DrawRectangle& thickness(int t) {
        if (t <= 0 && t != -1)
            throw ParameterError{"thickness", "must be positive or -1 (fill)", "DrawRectangle"};
        thickness_ = t;
        return *this;
    }

    Image<BGR> operator()(Image<BGR> img) const;

private:
    cv::Rect   rect_;
    cv::Scalar color_{0, 255, 0};
    int        thickness_{1};
};

} // namespace improc::core
