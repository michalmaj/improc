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
 * @code static_assert(BGRFormat<BGR>); @endcode
 */
template<typename F>
concept BGRFormat = AnyFormat<F> && std::same_as<F, BGR>;

/**
 * @brief Satisfied only by `Gray`.
 * @code static_assert(GrayFormat<Gray>); @endcode
 */
template<typename F>
concept GrayFormat = AnyFormat<F> && std::same_as<F, Gray>;

/**
 * @brief Satisfied by any format with more than one channel (BGR, BGRA, HSV, Float32C3).
 * @code static_assert(MultiChannelFormat<BGR>); @endcode
 */
template<typename F>
concept MultiChannelFormat = AnyFormat<F> && (FormatTraits<F>::channels > 1);

} // namespace improc::core
