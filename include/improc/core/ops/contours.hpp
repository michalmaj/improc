// include/improc/core/ops/contours.hpp
#pragma once
#include <vector>
#include <cstddef>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Result of a contour-finding operation.
 *
 * Plain struct with public members. Use `size()`, `area()`, `perimeter()`,
 * and `bounding_rect()` as convenience accessors.
 *
 * @code
 * ContourSet cs = gray | FindContours{};
 * for (std::size_t i = 0; i < cs.size(); ++i)
 *     std::cout << cs.area(i) << '\n';
 * @endcode
 */
struct ContourSet {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i>              hierarchy;

    std::size_t size()       const { return contours.size(); }
    bool        empty()      const { return contours.empty(); }

    /// @brief Area of contour `i` via `cv::contourArea`.
    double   area(std::size_t i) const { return cv::contourArea(contours.at(i)); }
    /// @brief Perimeter of contour `i` via `cv::arcLength` (closed=true).
    double   perimeter(std::size_t i) const { return cv::arcLength(contours.at(i), true); }
    /// @brief Bounding rectangle of contour `i` via `cv::boundingRect`.
    cv::Rect bounding_rect(std::size_t i) const { return cv::boundingRect(contours.at(i)); }
};

/**
 * @brief Pipeline op: finds contours in a binary `Image<Gray>`.
 *
 * Returns `ContourSet`. Wraps `cv::findContours`.
 *
 * @code
 * ContourSet cs = binary_gray | FindContours{}.mode(FindContours::Mode::Tree);
 * @endcode
 */
struct FindContours {
    enum class Mode   { External, List, CComp, Tree };
    enum class Method { None, Simple, TehChin };

    FindContours& mode(Mode m)     { mode_   = m; return *this; }
    FindContours& method(Method m) { method_ = m; return *this; }

    ContourSet operator()(Image<Gray> img) const;

private:
    Mode   mode_   = Mode::External;
    Method method_ = Method::Simple;
};

struct ConvexHull {
    std::vector<cv::Point> operator()(const std::vector<cv::Point>& contour) const;
};

struct ApproxPolyDP {
    ApproxPolyDP& epsilon(double e) { epsilon_ = e; return *this; }
    ApproxPolyDP& closed(bool c)    { closed_  = c; return *this; }

    std::vector<cv::Point> operator()(const std::vector<cv::Point>& contour) const;

private:
    double epsilon_ = 3.0;
    bool   closed_  = true;
};

struct MinAreaRect {
    cv::RotatedRect operator()(const std::vector<cv::Point>& contour) const;
};

struct BoundingRect {
    cv::Rect operator()(const std::vector<cv::Point>& contour) const;
};

} // namespace improc::core
