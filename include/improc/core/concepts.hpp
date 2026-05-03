// include/improc/core/concepts.hpp
#pragma once

#include <concepts>
#include "improc/core/format_traits.hpp"

namespace improc::core {

/**
 * @brief Satisfied by any type `F` with a complete `FormatTraits<F>` specialization.
 *
 * Requires `cv_type` (int), `channels` (int), and `is_float` (bool).
 * All pipeline ops are constrained by this concept.
 *
 * @code
 * static_assert(AnyFormat<BGR>);
 * static_assert(!AnyFormat<int>);
 * @endcode
 */
template<typename F>
concept AnyFormat = requires {
    { FormatTraits<F>::cv_type }  -> std::convertible_to<int>;
    { FormatTraits<F>::channels } -> std::convertible_to<int>;
    { FormatTraits<F>::is_float } -> std::convertible_to<bool>;
};

/**
 * @brief Satisfied only by `BGR`.
 *
 * Use to constrain ops that are only meaningful on BGR input,
 * such as `ToGray`, `ToFloat32C3`, `ToHSV`, or `AlphaBlend`.
 * @code static_assert(BGRFormat<BGR>); @endcode
 */
template<typename F>
concept BGRFormat = AnyFormat<F> && std::same_as<F, BGR>;

/**
 * @brief Satisfied only by `Gray`.
 *
 * Use to constrain ops that are only meaningful on single-channel input,
 * such as `ToFloat32` or `ToBGR` (from Gray).
 * @code static_assert(GrayFormat<Gray>); @endcode
 */
template<typename F>
concept GrayFormat = AnyFormat<F> && std::same_as<F, Gray>;

/**
 * @brief Satisfied by any format with more than one channel (BGR, BGRA, HSV, LAB, YCrCb, Float32C3).
 * @code static_assert(MultiChannelFormat<BGR>); @endcode
 */
template<typename F>
concept MultiChannelFormat = AnyFormat<F> && (FormatTraits<F>::channels > 1);

/**
 * @brief Satisfied by any non-floating-point format (Gray, BGR, BGRA, HSV, LAB, YCrCb).
 *
 * Use to constrain ops where bitwise operations or integer arithmetic are applied,
 * such as `Invert`.
 * @code static_assert(IntegerFormat<BGR>); static_assert(!IntegerFormat<Float32>); @endcode
 */
template<typename F>
concept IntegerFormat = AnyFormat<F> && !FormatTraits<F>::is_float;

} // namespace improc::core
