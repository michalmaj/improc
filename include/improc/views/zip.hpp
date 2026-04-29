// include/improc/views/zip.hpp
#pragma once

#include <utility>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/views/collection.hpp"

namespace improc::views {

using improc::core::Image;
using improc::core::AnyFormat;

// ── ZipView<Inner1, Inner2> ───────────────────────────────────────────────────

/// Lazy view that pairs elements from two sources element-by-element.
/// Stops when either source is exhausted (stops at the shorter).
template<typename Inner1, typename Inner2>
class ZipView {
    using iter1 = decltype(std::declval<Inner1>().begin());
    using iter2 = decltype(std::declval<Inner2>().begin());
    using elem1 = std::decay_t<decltype(*std::declval<iter1>())>;
    using elem2 = std::decay_t<decltype(*std::declval<iter2>())>;

public:
    ZipView(Inner1 v1, Inner2 v2) : v1_(std::move(v1)), v2_(std::move(v2)) {}

    struct iterator {
        iter1 it1_, end1_;
        iter2 it2_, end2_;

        bool at_end() const { return it1_ == end1_ || it2_ == end2_; }

        std::pair<elem1, elem2> operator*() const { return {*it1_, *it2_}; }

        iterator& operator++() { ++it1_; ++it2_; return *this; }

        bool operator==(const iterator& o) const {
            if (at_end() && o.at_end()) return true;
            return it1_ == o.it1_ && it2_ == o.it2_;
        }
        bool operator!=(const iterator& o) const { return !(*this == o); }
    };

    iterator begin() const {
        return {v1_.begin(), v1_.end(), v2_.begin(), v2_.end()};
    }
    iterator end() const {
        return {v1_.end(), v1_.end(), v2_.end(), v2_.end()};
    }

private:
    Inner1 v1_;
    Inner2 v2_;
};

// ── zip() factory — 4 overloads for vector/view combinations ─────────────────

/// Both arguments are non-vector views (already wrapped).
template<typename V1, typename V2>
auto zip(V1 v1, V2 v2) -> ZipView<V1, V2>
{
    return {std::move(v1), std::move(v2)};
}

/// v1 is a const vector (wraps in VectorView), v2 is a view.
template<AnyFormat F1, typename V2>
auto zip(const std::vector<Image<F1>>& v1, V2 v2)
    -> ZipView<VectorView<F1>, V2>
{
    return {VectorView<F1>{v1}, std::move(v2)};
}

/// v1 is a view, v2 is a const vector.
template<typename V1, AnyFormat F2>
auto zip(V1 v1, const std::vector<Image<F2>>& v2)
    -> ZipView<V1, VectorView<F2>>
{
    return {std::move(v1), VectorView<F2>{v2}};
}

/// Both arguments are const vectors.
template<AnyFormat F1, AnyFormat F2>
auto zip(const std::vector<Image<F1>>& v1, const std::vector<Image<F2>>& v2)
    -> ZipView<VectorView<F1>, VectorView<F2>>
{
    return {VectorView<F1>{v1}, VectorView<F2>{v2}};
}

/// Prevent dangling: temporary vectors cannot be sources.
template<AnyFormat F, typename V2>
auto zip(std::vector<Image<F>>&&, V2) -> void = delete;

template<typename V1, AnyFormat F2>
auto zip(V1, std::vector<Image<F2>>&&) -> void = delete;

} // namespace improc::views
