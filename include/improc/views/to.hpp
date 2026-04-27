// include/improc/views/to.hpp
#pragma once

#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/views/transform.hpp"

namespace improc::views {

using improc::core::Image;
using improc::core::AnyFormat;

// ── ToTag ─────────────────────────────────────────────────────────────────────

/// Tag type returned by views::to<Image<F>>(). Triggers materialization when piped.
template<typename T>
struct ToTag {};

/// Factory: views::to<Image<BGR>>()
template<typename T>
auto to() -> ToTag<T> {
    return ToTag<T>{};
}

// ── Materializing operator| ───────────────────────────────────────────────────

/// TransformView<F, Ops...> | views::to<Image<F>>()  →  Image<F>
/// Format mismatch (e.g. to<Image<Gray>>() on a BGR view) is a compile error:
/// no matching operator| exists for mismatched formats.
template<AnyFormat F, typename... Ops>
auto operator|(TransformView<F, Ops...> view, ToTag<Image<F>>) -> Image<F> {
    return view.eval();
}

} // namespace improc::views
