// include/improc/core/ops/hough.hpp
#pragma once

#include <vector>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

/**
 * @brief Detects line segments in a grayscale (edge) image using the probabilistic Hough transform.
 *
 * @code
 * auto lines = HoughLinesP().threshold(50).min_line_length(30.0)(edge_img);
 * @endcode
 */
struct HoughLinesP {
    /// @brief Sets the distance resolution of the accumulator in pixels (default: 1.0).
    HoughLinesP& rho(double r);
    /// @brief Sets the angle resolution of the accumulator in radians (default: 1 degree).
    HoughLinesP& theta(double t);
    /// @brief Sets the accumulator threshold — only lines with enough votes are returned (default: 80).
    HoughLinesP& threshold(int t);
    /// @brief Sets the minimum accepted line length in pixels (default: 0.0).
    HoughLinesP& min_line_length(double l);
    /// @brief Sets the maximum allowed gap between collinear points to link them (default: 0.0).
    HoughLinesP& max_line_gap(double g);

    /// @return Detected line segments as cv::Vec4i (x1, y1, x2, y2).
    std::vector<cv::Vec4i> operator()(const Image<Gray>& img) const;

private:
    double rho_             = 1.0;
    double theta_           = CV_PI / 180.0;
    int    threshold_       = 80;
    double min_line_length_ = 0.0;
    double max_line_gap_    = 0.0;
};

/**
 * @brief Detects circles in a grayscale image using the Hough gradient method.
 *
 * @code
 * auto circles = HoughCircles().min_dist(20.0).param2(30.0)(gray_img);
 * @endcode
 */
struct HoughCircles {
    /// @brief Sets the minimum distance between detected circle centres (default: 20.0).
    HoughCircles& min_dist(double d);
    /// @brief Sets the higher threshold for the Canny edge detector (default: 100.0).
    HoughCircles& param1(double p);
    /// @brief Sets the accumulator threshold for circle centres (default: 30.0).
    HoughCircles& param2(double p);
    /// @brief Sets the minimum circle radius in pixels (default: 0).
    HoughCircles& min_radius(int r);
    /// @brief Sets the maximum circle radius in pixels; 0 means no upper limit (default: 0).
    HoughCircles& max_radius(int r);

    /// @return Detected circles as cv::Vec3f (cx, cy, radius).
    std::vector<cv::Vec3f> operator()(const Image<Gray>& img) const;

private:
    double min_dist_   = 20.0;
    double param1_     = 100.0;
    double param2_     = 30.0;
    int    min_radius_ = 0;
    int    max_radius_ = 0;
};

} // namespace improc::core
