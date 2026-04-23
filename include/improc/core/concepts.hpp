// include/improc/core/concepts.hpp
#pragma once

#include <concepts>
#include "improc/core/format_traits.hpp"

namespace improc::core {

template<typename F>
concept AnyFormat = requires {
    { FormatTraits<F>::cv_type }  -> std::convertible_to<int>;
    { FormatTraits<F>::channels } -> std::convertible_to<int>;
    { FormatTraits<F>::is_float } -> std::convertible_to<bool>;
};

template<typename F>
concept BGRFormat = AnyFormat<F> && std::same_as<F, BGR>;

template<typename F>
concept GrayFormat = AnyFormat<F> && std::same_as<F, Gray>;

template<typename F>
concept MultiChannelFormat = AnyFormat<F> && (FormatTraits<F>::channels > 1);

} // namespace improc::core
