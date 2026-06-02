// include/improc/version.hpp
#pragma once

#define IMPROC_VERSION_MAJOR 0
#define IMPROC_VERSION_MINOR 18
#define IMPROC_VERSION_PATCH 0

/// Encoded as MAJOR*10000 + MINOR*100 + PATCH — use for compile-time comparisons:
/// @code
/// #if IMPROC_VERSION >= 1800  // >= 0.18.0
/// @endcode
#define IMPROC_VERSION (IMPROC_VERSION_MAJOR * 10000 + IMPROC_VERSION_MINOR * 100 + IMPROC_VERSION_PATCH)

#define IMPROC_VERSION_STRING "0.18.0"

namespace improc {
/// Returns the library version string, e.g. "0.18.0".
inline constexpr const char* version_string() noexcept { return IMPROC_VERSION_STRING; }
} // namespace improc
