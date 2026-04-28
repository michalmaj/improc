// include/improc/views/collection.hpp
#pragma once

#include <vector>
#include <iterator>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/views/transform.hpp"

namespace improc::views {

using improc::core::Image;
using improc::core::AnyFormat;

// ── ImageCollectionView concept ───────────────────────────────────────────────

/// Satisfied by any type that is iterable and whose elements are Image<F>.
template<typename V, typename F>
concept ImageCollectionView = AnyFormat<F> && requires(V v) {
    { v.begin() };
    { v.end() };
    { *v.begin() } -> std::convertible_to<Image<F>>;
};

// ── VectorView<F> ─────────────────────────────────────────────────────────────

/// Lazy base view over a const std::vector<Image<F>>.
/// Holds a non-owning reference — the source vector must outlive the view.
template<AnyFormat F>
class VectorView {
public:
    using value_type = Image<F>;

    explicit VectorView(const std::vector<Image<F>>& vec) : vec_(&vec) {}

    auto begin() const { return vec_->begin(); }
    auto end()   const { return vec_->end(); }

private:
    const std::vector<Image<F>>* vec_;
};

// ── CollectionTransformView<Inner, Op> ────────────────────────────────────────

/// Lazy view that applies Op to each element of Inner on dereference.
template<typename Inner, typename Op>
class CollectionTransformView {
public:
    using inner_iter = decltype(std::declval<Inner>().begin());

    CollectionTransformView(Inner inner, Op op)
        : inner_(std::move(inner)), op_(std::move(op)) {}

    struct iterator {
        inner_iter it_;
        const Op*  op_;

        auto operator*() const { return (*op_)(*it_); }
        iterator& operator++() { ++it_; return *this; }
        bool operator==(const iterator& o) const { return it_ == o.it_; }
        bool operator!=(const iterator& o) const { return it_ != o.it_; }
    };

    iterator begin() const { return {inner_.begin(), &op_}; }
    iterator end()   const { return {inner_.end(),   &op_}; }

private:
    Inner inner_;
    Op    op_;
};

// ── operator| overloads for collections ──────────────────────────────────────

/// std::vector<Image<F>> | views::transform(op)  →  CollectionTransformView
template<AnyFormat F, typename Op>
auto operator|(const std::vector<Image<F>>& vec, TransformAdapter<Op> adapter)
    -> CollectionTransformView<VectorView<F>, Op>
{
    return {VectorView<F>{vec}, std::move(adapter.op)};
}

/// CollectionTransformView | views::transform(op2)  →  wrap in another layer
template<typename Inner, typename Op1, typename Op2>
auto operator|(CollectionTransformView<Inner, Op1> view, TransformAdapter<Op2> adapter)
    -> CollectionTransformView<CollectionTransformView<Inner, Op1>, Op2>
{
    return {std::move(view), std::move(adapter.op)};
}

} // namespace improc::views
