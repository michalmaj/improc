// include/improc/core/ops/normalize.hpp
#pragma once

#include <stdexcept>
#include "improc/core/image.hpp"

namespace improc::core {

/**
 * @brief Normalizes pixel values to the [0, 1] range using min-max scaling.
 *
 * Only operates on float images (`Image<Float32>`, `Image<Float32C3>`).
 * If the image is uniform (min == max), all output values are 0.
 *
 * @code
 * Image<Float32C3> norm = float_img | Normalize{};
 * @endcode
 */
struct Normalize {
    /// @brief Applies min-max normalization to img, scaling values to [0, 1].
    Image<Float32>   operator()(Image<Float32>   img) const;
    /// @brief Applies min-max normalization to img, scaling values to [0, 1].
    Image<Float32C3> operator()(Image<Float32C3> img) const;
};

/**
 * @brief Scales pixel values to a specified [min_val, max_val] range.
 *
 * Only operates on float images (`Image<Float32>`, `Image<Float32C3>`).
 * If the image is uniform (min == max), all output values are 0.
 *
 * @code
 * Image<Float32C3> scaled = float_img | NormalizeTo{0.0f, 1.0f};
 * @endcode
 */
struct NormalizeTo {
    /**
     * @brief Constructs NormalizeTo with output range [min_val, max_val].
     * @throws improc::ParameterError if min_val >= max_val.
     */
    explicit NormalizeTo(float min, float max);
    /// @brief Scales pixel values of img to [min_val, max_val].
    Image<Float32>   operator()(Image<Float32>   img) const;
    /// @brief Scales pixel values of img to [min_val, max_val].
    Image<Float32C3> operator()(Image<Float32C3> img) const;
private:
    float min_, max_;
};

/**
 * @brief Standardizes pixel values: `output = (input - mean) / std_dev`.
 *
 * Only operates on float images (`Image<Float32>`, `Image<Float32C3>`).
 *
 * @code
 * Image<Float32C3> std_img = float_img | Standardize{0.485f, 0.229f};
 * @endcode
 */
struct Standardize {
    /**
     * @brief Constructs Standardize with the given mean and standard deviation.
     * @throws improc::ParameterError if std_dev <= 0.
     */
    explicit Standardize(float mean, float std_dev);
    /// @brief Standardizes img by subtracting mean and dividing by std_dev.
    Image<Float32>   operator()(Image<Float32>   img) const;
    /// @brief Standardizes img by subtracting mean and dividing by std_dev.
    Image<Float32C3> operator()(Image<Float32C3> img) const;
private:
    float mean_, std_dev_;
};

} // namespace improc::core
