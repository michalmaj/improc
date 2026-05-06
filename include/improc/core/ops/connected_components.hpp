// include/improc/core/ops/connected_components.hpp
#pragma once
#include <stdexcept>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

/**
 * @brief Result of a connected-components labelling operation.
 *
 * Plain struct with public members. Label 0 is always the background.
 * Accessor methods are bounds-checked — invalid label throws `std::out_of_range`.
 *
 * @code
 * ComponentMap cm = binary | ConnectedComponents{};
 * for (int i = 1; i < cm.count(); ++i)
 *     std::cout << cm.area(i) << '\n';
 * @endcode
 */
struct ComponentMap {
    cv::Mat labels;        ///< CV_32S label matrix, same size as source
    cv::Mat stats;         ///< (num_labels × 5) int stats matrix (CC_STAT_* columns)
    cv::Mat centroids;     ///< (num_labels × 2) double centroid matrix
    int     num_labels{0}; ///< total label count including background (label 0)

    /// @brief Alias for `num_labels`.
    int count() const { return num_labels; }

    /// @brief Pixel area of label `label` (CC_STAT_AREA). Throws `std::out_of_range` if invalid.
    int area(int label) const;

    /// @brief Bounding rectangle of label `label`. Throws `std::out_of_range` if invalid.
    cv::Rect bounding_rect(int label) const;

    /// @brief Centroid of label `label`. Throws `std::out_of_range` if invalid.
    cv::Point2d centroid(int label) const;

    /// @brief Binary mask (CV_8U) for label `label` — 255 where labels==label, 0 elsewhere.
    /// Throws `std::out_of_range` if invalid.
    cv::Mat mask(int label) const;

private:
    void check(int label) const;
};

/**
 * @brief Pipeline op: labels connected components in a binary `Image<Gray>`.
 *
 * Returns `ComponentMap`. Wraps `cv::connectedComponentsWithStats`.
 * Label 0 is always the background.
 *
 * @code
 * ComponentMap cm = binary | ConnectedComponents{};
 * ComponentMap cm4 = binary | ConnectedComponents{}.connectivity(ConnectedComponents::Connectivity::Four);
 * @endcode
 */
struct ConnectedComponents {
    enum class Connectivity { Four = 4, Eight = 8 };

    ConnectedComponents& connectivity(Connectivity c) { conn_ = c; return *this; }

    ComponentMap operator()(Image<Gray> img) const;

private:
    Connectivity conn_ = Connectivity::Eight;
};

} // namespace improc::core
