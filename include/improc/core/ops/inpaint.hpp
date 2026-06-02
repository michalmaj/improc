// include/improc/core/ops/inpaint.hpp
#pragma once

#include <opencv2/photo.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

/// @brief Inpainting algorithm choices.
enum class InpaintMethod {
    TELEA, ///< Fast Marching Method (Telea 2004) — good for thin regions.
    NS,    ///< Navier-Stokes fluid dynamics method — better for large holes.
};

/**
 * @brief Restores damaged or masked regions of a BGR image using inpainting.
 *
 * @code
 * auto restored = Inpaint().radius(5.0).method(InpaintMethod::TELEA)(img, mask);
 * @endcode
 */
struct Inpaint {
    /// @brief Sets the inpainting neighbourhood radius in pixels (default: 3.0).
    Inpaint& radius(double r);
    /// @brief Sets the inpainting algorithm (default: InpaintMethod::TELEA).
    Inpaint& method(InpaintMethod m);

    /// @brief Inpaints regions where mask is non-zero.
    [[nodiscard]] Image<BGR> operator()(const Image<BGR>& img, const Image<Gray>& mask) const;

private:
    double radius_ = 3.0;
    InpaintMethod method_ = InpaintMethod::TELEA;
};

} // namespace improc::core
