// include/improc/version.hpp
#pragma once

#define IMPROC_VERSION_MAJOR 1
#define IMPROC_VERSION_MINOR 0
#define IMPROC_VERSION_PATCH 3

/// Encoded as MAJOR*10000 + MINOR*100 + PATCH — use for compile-time comparisons:
/// @code
/// #if IMPROC_VERSION >= 10000  // >= 1.0.0
/// @endcode
#define IMPROC_VERSION (IMPROC_VERSION_MAJOR * 10000 + IMPROC_VERSION_MINOR * 100 + IMPROC_VERSION_PATCH)

#define IMPROC_VERSION_STRING "1.0.3"

namespace improc {
/// Returns the library version string, e.g. "1.0.2". Defined in src/version.cpp — links a real symbol.
const char* version_string() noexcept;
} // namespace improc
