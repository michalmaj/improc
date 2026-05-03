/**
 * @brief Umbrella include for all `improc::core` pipeline ops.
 *
 * Including this header pulls in every op (Resize, Crop, CenterCrop, LetterBox, Flip, Rotate, Pad,
 * GaussianBlur, MedianBlur, Dilate, Erode, MorphOpen, MorphClose, MorphGradient, TopHat, BlackHat, Threshold, AdaptiveThreshold, InRange, Invert, CLAHE, GammaCorrection,
 * BilateralFilter, HistogramEqualization, NLMeansDenoising, UnsharpMask, SobelEdge, CannyEdge, LaplacianEdge, HarrisCorner, Normalize, NormalizeTo,
 * Standardize, ApplyMask, WarpAffine, WarpPerspective, ToGray, ToFloat32,
 * ToFloat32C3, ToHSV, ToLAB, ToYCrCb, ToBGR, Brightness, Contrast, WeightedBlend, AlphaBlend)
 * and the generic `operator|` pipeline dispatch.
 *
 * @code
 * #include "improc/core/pipeline.hpp"
 * using namespace improc::core;
 *
 * Image<BGR> result = img
 *     | Resize{}.width(224).height(224)
 *     | GaussianBlur{}.kernel_size(3)
 *     | Brightness{}.delta(20.0);
 * @endcode
 */
#pragma once

#include "improc/core/image.hpp"
#include "improc/core/convert.hpp"
#include "improc/core/ops/axis.hpp"
#include "improc/core/ops/resize.hpp"
#include "improc/core/ops/crop.hpp"
#include "improc/core/ops/center_crop.hpp"
#include "improc/core/ops/letter_box.hpp"
#include "improc/core/ops/flip.hpp"
#include "improc/core/ops/rotate.hpp"
#include "improc/core/ops/normalize.hpp"
#include "improc/core/ops/blur.hpp"
#include "improc/core/ops/morphology.hpp"
#include "improc/core/ops/threshold.hpp"
#include "improc/core/ops/adaptive_threshold.hpp"
#include "improc/core/ops/in_range.hpp"
#include "improc/core/ops/invert.hpp"
#include "improc/core/ops/pad.hpp"
#include "improc/core/ops/clahe.hpp"
#include "improc/core/ops/gamma.hpp"
#include "improc/core/ops/bilateral_filter.hpp"
#include "improc/core/ops/hist_eq.hpp"
#include "improc/core/ops/nlmeans.hpp"
#include "improc/core/ops/edge.hpp"
#include "improc/core/ops/homography.hpp"
#include "improc/core/ops/warp_affine.hpp"
#include "improc/core/ops/apply_mask.hpp"
#include "improc/core/ops/unsharp_mask.hpp"
#include "improc/core/ops/to_hsv.hpp"
#include "improc/core/ops/to_bgr.hpp"
#include "improc/core/ops/to_lab.hpp"
#include "improc/core/ops/to_ycrcb.hpp"
#include "improc/core/ops/brightness.hpp"
#include "improc/core/ops/contrast.hpp"
#include "improc/core/ops/weighted_blend.hpp"
#include "improc/core/ops/alpha_blend.hpp"

namespace improc::core {

/**
 * @brief Generic pipeline dispatch operator.
 *
 * Passes `img` by value into `op`, enabling the fluent `img | Op{}` syntax.
 * Every pipeline op in `improc::core` uses this single overload.
 *
 * @tparam Format  Format tag of the source image.
 * @tparam Op      A callable that accepts `Image<SomeFormat>` and returns an image.
 * @return         Whatever `op` returns.
 * @param  img     Source image, passed by value (ownership moved into `op`).
 * @param  op      A pipeline op callable; must accept `Image<Format>` or a subtype.
 *
 * @code
 * Image<Gray> gray = bgr_img | ToGray{} | Brightness{}.delta(10.0);
 * @endcode
 */
template<AnyFormat Format, typename Op>
auto operator|(Image<Format> img, Op&& op) {
    return std::forward<Op>(op)(std::move(img));
}

/// @brief Pipeline op: converts a BGR image to single-channel Gray.
struct ToGray      { Image<Gray>      operator()(Image<BGR>  img) const; };

/// @brief Pipeline op: converts a Gray image to single-channel Float32 (values in [0, 1]).
struct ToFloat32   { Image<Float32>   operator()(Image<Gray> img) const; };

/// @brief Pipeline op: converts a BGR image to 3-channel Float32C3 (values in [0, 1]).
struct ToFloat32C3 { Image<Float32C3> operator()(Image<BGR>  img) const; };

} // namespace improc::core
