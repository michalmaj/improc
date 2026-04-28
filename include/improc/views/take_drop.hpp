// include/improc/views/take_drop.hpp
#pragma once

#include <cstddef>
#include <utility>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/views/collection.hpp"
#include "improc/views/filter.hpp"

namespace improc::views {

using improc::core::Image;
using improc::core::AnyFormat;

// ── TakeView<Inner> ───────────────────────────────────────────────────────────

/// Lazy view that stops after N elements.
template<typename Inner>
class TakeView {
public:
    using inner_iter = decltype(std::declval<Inner>().begin());

    TakeView(Inner inner, std::size_t n)
        : inner_(std::move(inner)), n_(n) {}

    struct iterator {
        inner_iter  it_;
        inner_iter  end_;
        std::size_t count_;
        std::size_t limit_;

        bool at_end() const { return count_ >= limit_ || it_ == end_; }

        auto operator*() const { return *it_; }

        iterator& operator++() {
            ++it_;
            ++count_;
            return *this;
        }

        bool operator==(const iterator& o) const {
            if (at_end() && o.at_end()) return true;
            return it_ == o.it_ && count_ == o.count_;
        }
        bool operator!=(const iterator& o) const { return !(*this == o); }
    };

    iterator begin() const { return {inner_.begin(), inner_.end(), 0, n_}; }
    iterator end()   const { return {inner_.end(),   inner_.end(), n_, n_}; }

private:
    Inner       inner_;
    std::size_t n_;
};

// ── DropView<Inner> ───────────────────────────────────────────────────────────

/// Lazy view that skips the first N elements.
template<typename Inner>
class DropView {
public:
    using inner_iter = decltype(std::declval<Inner>().begin());

    DropView(Inner inner, std::size_t n)
        : inner_(std::move(inner)), n_(n) {}

    inner_iter begin() const {
        auto it  = inner_.begin();
        auto end = inner_.end();
        for (std::size_t i = 0; i < n_ && it != end; ++i) ++it;
        return it;
    }
    inner_iter end() const { return inner_.end(); }

private:
    Inner       inner_;
    std::size_t n_;
};

// ── Adapters ──────────────────────────────────────────────────────────────────

struct TakeAdapter { std::size_t n; };
struct DropAdapter { std::size_t n; };

inline auto take(std::size_t n) -> TakeAdapter { return {n}; }
inline auto drop(std::size_t n) -> DropAdapter { return {n}; }

// ── operator| for take ────────────────────────────────────────────────────────

template<AnyFormat F>
auto operator|(const std::vector<Image<F>>& vec, TakeAdapter a)
    -> TakeView<VectorView<F>>
{
    return {VectorView<F>{vec}, a.n};
}

/// Prevents binding a temporary vector — VectorView stores a pointer; the source must outlive it.
template<AnyFormat F>
auto operator|(std::vector<Image<F>>&&, TakeAdapter) -> void = delete;

template<typename Inner, typename Op>
auto operator|(CollectionTransformView<Inner, Op> view, TakeAdapter a)
    -> TakeView<CollectionTransformView<Inner, Op>>
{
    return {std::move(view), a.n};
}

template<typename Inner, typename Pred>
auto operator|(FilterView<Inner, Pred> view, TakeAdapter a)
    -> TakeView<FilterView<Inner, Pred>>
{
    return {std::move(view), a.n};
}

template<typename Inner>
auto operator|(TakeView<Inner> view, TakeAdapter a)
    -> TakeView<TakeView<Inner>>
{
    return {std::move(view), a.n};
}

template<typename Inner>
auto operator|(DropView<Inner> view, TakeAdapter a)
    -> TakeView<DropView<Inner>>
{
    return {std::move(view), a.n};
}

// ── operator| for drop ────────────────────────────────────────────────────────

template<AnyFormat F>
auto operator|(const std::vector<Image<F>>& vec, DropAdapter a)
    -> DropView<VectorView<F>>
{
    return {VectorView<F>{vec}, a.n};
}

template<AnyFormat F>
auto operator|(std::vector<Image<F>>&&, DropAdapter) -> void = delete;

template<typename Inner, typename Op>
auto operator|(CollectionTransformView<Inner, Op> view, DropAdapter a)
    -> DropView<CollectionTransformView<Inner, Op>>
{
    return {std::move(view), a.n};
}

template<typename Inner, typename Pred>
auto operator|(FilterView<Inner, Pred> view, DropAdapter a)
    -> DropView<FilterView<Inner, Pred>>
{
    return {std::move(view), a.n};
}

template<typename Inner>
auto operator|(DropView<Inner> view, DropAdapter a)
    -> DropView<DropView<Inner>>
{
    return {std::move(view), a.n};
}

template<typename Inner>
auto operator|(TakeView<Inner> view, DropAdapter a)
    -> DropView<TakeView<Inner>>
{
    return {std::move(view), a.n};
}

} // namespace improc::views
