// include/improc/views/enumerate.hpp
#pragma once

#include <cstddef>
#include <utility>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/views/collection.hpp"
#include "improc/views/filter.hpp"
#include "improc/views/take_drop.hpp"

namespace improc::views {

using improc::core::Image;
using improc::core::AnyFormat;

// ── EnumerateView<Inner> ──────────────────────────────────────────────────────

/// Lazy view that pairs each element with a zero-based std::size_t index.
template<typename Inner>
class EnumerateView {
public:
    using inner_iter = decltype(std::declval<Inner>().begin());
    using elem_t     = std::decay_t<decltype(*std::declval<inner_iter>())>;

    explicit EnumerateView(Inner inner) : inner_(std::move(inner)) {}

    struct iterator {
        inner_iter  it_;
        inner_iter  end_;
        std::size_t idx_;

        std::pair<std::size_t, elem_t> operator*() const { return {idx_, *it_}; }

        iterator& operator++() { ++it_; ++idx_; return *this; }

        bool operator==(const iterator& o) const { return it_ == o.it_; }
        bool operator!=(const iterator& o) const { return it_ != o.it_; }
    };

    iterator begin() const { return {inner_.begin(), inner_.end(), 0}; }
    iterator end()   const { return {inner_.end(),   inner_.end(), 0}; }

private:
    Inner inner_;
};

// ── EnumerateTag — tag object, used without () ────────────────────────────────

struct EnumerateTag {};

/// views::enumerate  — pair each element with its zero-based index (no parentheses)
inline constexpr EnumerateTag enumerate{};

// ── operator| overloads ───────────────────────────────────────────────────────

template<AnyFormat F>
auto operator|(const std::vector<Image<F>>& vec, EnumerateTag)
    -> EnumerateView<VectorView<F>>
{
    return EnumerateView<VectorView<F>>{VectorView<F>{vec}};
}

template<AnyFormat F>
auto operator|(std::vector<Image<F>>&&, EnumerateTag) -> void = delete;

template<typename Inner, typename Op>
auto operator|(CollectionTransformView<Inner, Op> view, EnumerateTag)
    -> EnumerateView<CollectionTransformView<Inner, Op>>
{
    return EnumerateView<CollectionTransformView<Inner, Op>>{std::move(view)};
}

template<typename Inner, typename Pred>
auto operator|(FilterView<Inner, Pred> view, EnumerateTag)
    -> EnumerateView<FilterView<Inner, Pred>>
{
    return EnumerateView<FilterView<Inner, Pred>>{std::move(view)};
}

template<typename Inner>
auto operator|(TakeView<Inner> view, EnumerateTag)
    -> EnumerateView<TakeView<Inner>>
{
    return EnumerateView<TakeView<Inner>>{std::move(view)};
}

template<typename Inner>
auto operator|(DropView<Inner> view, EnumerateTag)
    -> EnumerateView<DropView<Inner>>
{
    return EnumerateView<DropView<Inner>>{std::move(view)};
}

} // namespace improc::views
