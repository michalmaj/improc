// include/improc/views/filter.hpp
#pragma once

#include <utility>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/views/collection.hpp"

namespace improc::views {

using improc::core::Image;
using improc::core::AnyFormat;

// ── FilterView<Inner, Pred> ───────────────────────────────────────────────────

/// Lazy view that skips elements for which Pred returns false.
template<typename Inner, typename Pred>
class FilterView {
public:
    using inner_iter = decltype(std::declval<Inner>().begin());
    using inner_end  = decltype(std::declval<Inner>().end());

    FilterView(Inner inner, Pred pred)
        : inner_(std::move(inner)), pred_(std::move(pred)) {}

    struct iterator {
        inner_iter  it_;
        inner_end   end_;
        const Pred* pred_;

        void advance() {
            while (it_ != end_ && !(*pred_)(*it_)) ++it_;
        }

        iterator(inner_iter it, inner_end end, const Pred* pred)
            : it_(it), end_(end), pred_(pred) { advance(); }

        auto operator*() const { return *it_; }

        iterator& operator++() {
            ++it_;
            advance();
            return *this;
        }

        bool operator==(const iterator& o) const { return it_ == o.it_; }
        bool operator!=(const iterator& o) const { return it_ != o.it_; }
    };

    iterator begin() const { return {inner_.begin(), inner_.end(), &pred_}; }
    iterator end()   const { return {inner_.end(),   inner_.end(), &pred_}; }

private:
    Inner inner_;
    Pred  pred_;
};

// ── FilterAdapter ─────────────────────────────────────────────────────────────

template<typename Pred>
struct FilterAdapter { Pred pred; };

template<typename Pred>
auto filter(Pred pred) -> FilterAdapter<Pred> {
    return FilterAdapter<Pred>{std::move(pred)};
}

// ── operator| overloads ───────────────────────────────────────────────────────

/// std::vector<Image<F>> | views::filter(pred)
template<AnyFormat F, typename Pred>
auto operator|(const std::vector<Image<F>>& vec, FilterAdapter<Pred> adapter)
    -> FilterView<VectorView<F>, Pred>
{
    return {VectorView<F>{vec}, std::move(adapter.pred)};
}

/// Prevents binding a temporary vector — VectorView stores a pointer; the source must outlive it.
template<AnyFormat F, typename Pred>
auto operator|(std::vector<Image<F>>&&, FilterAdapter<Pred>) -> void = delete;

/// CollectionTransformView | views::filter(pred)
template<typename Inner, typename Op, typename Pred>
auto operator|(CollectionTransformView<Inner, Op> view, FilterAdapter<Pred> adapter)
    -> FilterView<CollectionTransformView<Inner, Op>, Pred>
{
    return {std::move(view), std::move(adapter.pred)};
}

/// FilterView | views::filter(pred) — chain two filters
template<typename Inner, typename Pred1, typename Pred2>
auto operator|(FilterView<Inner, Pred1> view, FilterAdapter<Pred2> adapter)
    -> FilterView<FilterView<Inner, Pred1>, Pred2>
{
    return {std::move(view), std::move(adapter.pred)};
}

} // namespace improc::views
