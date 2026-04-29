// include/improc/views/batch.hpp
#pragma once

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/views/collection.hpp"
#include "improc/views/filter.hpp"
#include "improc/views/take_drop.hpp"

namespace improc::views {

using improc::core::Image;
using improc::core::AnyFormat;

// ── BatchView<Inner> ──────────────────────────────────────────────────────────

/// Lazy view that groups consecutive elements of Inner into std::vector chunks.
/// The last chunk may be smaller than n if the source has fewer remaining elements.
template<typename Inner>
class BatchView {
public:
    using inner_iter = decltype(std::declval<Inner>().begin());
    using elem_t     = std::decay_t<decltype(*std::declval<inner_iter>())>;

    BatchView(Inner inner, std::size_t n)
        : inner_(std::move(inner)), n_(n) {}

    struct iterator {
        inner_iter  it_;
        inner_iter  end_;
        std::size_t n_;
        std::optional<std::vector<elem_t>> current_;

        void load() {
            if (it_ == end_) { current_ = std::nullopt; return; }
            std::vector<elem_t> buf;
            for (std::size_t i = 0; i < n_ && it_ != end_; ++i, ++it_)
                buf.push_back(*it_);
            current_ = std::move(buf);
        }

        const std::vector<elem_t>& operator*() const { return current_.value(); }

        iterator& operator++() { load(); return *this; }

        bool operator==(const iterator& o) const {
            return !current_.has_value() && !o.current_.has_value();
        }
        bool operator!=(const iterator& o) const { return !(*this == o); }
    };

    iterator begin() const {
        iterator it{inner_.begin(), inner_.end(), n_, std::nullopt};
        it.load();
        return it;
    }
    iterator end() const {
        auto e = inner_.end();
        return {e, e, n_, std::nullopt};
    }

private:
    Inner       inner_;
    std::size_t n_;
};

// ── BatchAdapter ──────────────────────────────────────────────────────────────

struct BatchAdapter { std::size_t n; };

inline auto batch(std::size_t n) -> BatchAdapter { return {n}; }

// ── operator| overloads ───────────────────────────────────────────────────────

template<AnyFormat F>
auto operator|(const std::vector<Image<F>>& vec, BatchAdapter a)
    -> BatchView<VectorView<F>>
{
    return {VectorView<F>{vec}, a.n};
}

template<AnyFormat F>
auto operator|(std::vector<Image<F>>&&, BatchAdapter) -> void = delete;

template<typename Inner, typename Op>
auto operator|(CollectionTransformView<Inner, Op> view, BatchAdapter a)
    -> BatchView<CollectionTransformView<Inner, Op>>
{
    return {std::move(view), a.n};
}

template<typename Inner, typename Pred>
auto operator|(FilterView<Inner, Pred> view, BatchAdapter a)
    -> BatchView<FilterView<Inner, Pred>>
{
    return {std::move(view), a.n};
}

template<typename Inner>
auto operator|(TakeView<Inner> view, BatchAdapter a)
    -> BatchView<TakeView<Inner>>
{
    return {std::move(view), a.n};
}

template<typename Inner>
auto operator|(DropView<Inner> view, BatchAdapter a)
    -> BatchView<DropView<Inner>>
{
    return {std::move(view), a.n};
}

} // namespace improc::views
