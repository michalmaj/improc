#pragma once

#include <tuple>
#include <utility>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"

namespace improc::views {

using improc::core::Image;
using improc::core::AnyFormat;

// ── TransformView ─────────────────────────────────────────────────────────────

/// Lazy view: holds a source Image<F> and a tuple of ops to apply on eval().
/// None of the ops execute until operator| with ToImageTag is called.
template<AnyFormat F, typename... Ops>
class TransformView {
public:
    Image<F>           source_;   // public for tuple_cat chaining operator|
    std::tuple<Ops...> ops_;

    TransformView(Image<F> src, std::tuple<Ops...> ops)
        : source_(std::move(src)), ops_(std::move(ops)) {}

    /// Apply all ops left-to-right on the source. Called by views::to<>.
    Image<F> eval() const {
        Image<F> result = source_.clone();
        std::apply([&result](const auto&... ops) {
            ((result = ops(result)), ...);
        }, ops_);
        return result;
    }
};

// ── TransformAdapter ──────────────────────────────────────────────────────────

/// Lightweight tag returned by views::transform(op). No work done at construction.
template<typename Op>
struct TransformAdapter {
    Op op;
};

/// Factory: views::transform(Resize{}.width(224).height(224))
template<typename Op>
auto transform(Op op) -> TransformAdapter<Op> {
    return TransformAdapter<Op>{std::move(op)};
}

// ── operator| overloads ───────────────────────────────────────────────────────

/// Image<F> | views::transform(op)  →  TransformView<F, Op>
template<AnyFormat F, typename Op>
auto operator|(Image<F> img, TransformAdapter<Op> adapter)
    -> TransformView<F, Op>
{
    return TransformView<F, Op>{
        std::move(img),
        std::make_tuple(std::move(adapter.op))
    };
}

/// TransformView<F, Ops...> | views::transform(op)
/// Accumulates op into the tuple — no evaluation until views::to<>().
template<AnyFormat F, typename... Ops, typename Op>
auto operator|(TransformView<F, Ops...> view, TransformAdapter<Op> adapter)
    -> TransformView<F, Ops..., Op>
{
    auto new_ops = std::tuple_cat(
        std::move(view.ops_),
        std::make_tuple(std::move(adapter.op))
    );
    return TransformView<F, Ops..., Op>{
        std::move(view.source_),
        std::move(new_ops)
    };
}

} // namespace improc::views
