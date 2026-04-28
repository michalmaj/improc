// include/improc/views/to.hpp
#pragma once

#include <vector>

#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/views/transform.hpp"
#include "improc/views/collection.hpp"

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

// ── Collection materializer ───────────────────────────────────────────────────

/// Any iterable collection view | views::to<std::vector<Image<F>>>()
/// Works for CollectionTransformView, FilterView, TakeView, DropView and any
/// future type satisfying ImageCollectionView<V, F>.
template<AnyFormat F, typename V>
    requires ImageCollectionView<V, F>
auto operator|(V view, ToTag<std::vector<Image<F>>>) -> std::vector<Image<F>> {
    std::vector<Image<F>> result;
    for (auto img : view)
        result.push_back(std::move(img));
    return result;
}

} // namespace improc::views
