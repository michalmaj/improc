// include/improc/views/video_source.hpp
#pragma once

#include <optional>
#include <utility>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/io/video_reader.hpp"
#include "improc/views/collection.hpp"
#include "improc/views/filter.hpp"
#include "improc/views/take_drop.hpp"

namespace improc::views {

using improc::core::Image;
using improc::core::BGR;

// ── VideoView ─────────────────────────────────────────────────────────────────

/// Lazy single-pass view over a VideoReader.
/// Non-owning: the VideoReader must outlive this view and all iterators obtained from it.
/// Each call to begin() reads the next frame from the VideoReader — do not call begin() twice.
class VideoView {
public:
    explicit VideoView(improc::io::VideoReader& reader) : reader_(&reader) {}

    struct iterator {
        improc::io::VideoReader*  reader_;
        std::optional<Image<BGR>> current_;

        explicit iterator(improc::io::VideoReader* r) : reader_(r) {
            if (r) current_ = r->next();
        }
        iterator() : reader_(nullptr) {}  // end sentinel

        Image<BGR> operator*() const { return *current_; }

        iterator& operator++() {
            current_ = reader_ ? reader_->next() : std::nullopt;
            return *this;
        }

        bool operator==(const iterator& o) const {
            return !current_.has_value() && !o.current_.has_value();
        }
        bool operator!=(const iterator& o) const { return !(*this == o); }
    };

    /// Reads the next available frame from the VideoReader as the starting point.
    /// Side-effects the external VideoReader — do not call more than once.
    iterator begin() const { return iterator{reader_}; }
    /// Returns an end sentinel (empty iterator, no VideoReader call).
    iterator end()   const { return iterator{};        }

private:
    improc::io::VideoReader* reader_;
};

// ── operator| overloads ───────────────────────────────────────────────────────

/// VideoView | views::transform(op)  →  CollectionTransformView<VideoView, Op>
template<typename Op>
auto operator|(VideoView view, TransformAdapter<Op> adapter)
    -> CollectionTransformView<VideoView, Op>
{
    return {std::move(view), std::move(adapter.op)};
}

/// VideoView | views::filter(pred)  →  FilterView<VideoView, Pred>
template<typename Pred>
auto operator|(VideoView view, FilterAdapter<Pred> adapter)
    -> FilterView<VideoView, Pred>
{
    return {std::move(view), std::move(adapter.pred)};
}

/// VideoView | views::take(n)  →  TakeView<VideoView>
inline auto operator|(VideoView view, TakeAdapter a)
    -> TakeView<VideoView>
{
    return {std::move(view), a.n};
}

/// VideoView | views::drop(n)  →  DropView<VideoView>
inline auto operator|(VideoView view, DropAdapter a)
    -> DropView<VideoView>
{
    return {std::move(view), a.n};
}

} // namespace improc::views
