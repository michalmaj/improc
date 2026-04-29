// include/improc/views/from_dir.hpp
#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"
#include "improc/io/image_io.hpp"
#include "improc/views/collection.hpp"
#include "improc/views/filter.hpp"
#include "improc/views/take_drop.hpp"

namespace improc::views {

using improc::core::Image;
using improc::core::BGR;

// ── DirView ───────────────────────────────────────────────────────────────────

/// Lazy view over image files in a directory filtered by extension.
/// Extension matching is case-insensitive (e.g. ".PNG" matches ".png").
/// Throws FileNotFoundError at construction if the directory does not exist.
/// Failed imread calls are silently skipped (logged to std::cerr).
/// Note: begin() performs an O(n) directory scan to collect matching paths.
/// Images are then loaded lazily one at a time on iterator dereference.
class DirView {
public:
    DirView(std::filesystem::path dir, std::vector<std::string> exts)
        : dir_(std::move(dir)), exts_(std::move(exts))
    {
        if (!std::filesystem::exists(dir_))
            throw improc::FileNotFoundError{dir_};
    }

    struct iterator {
        std::vector<std::filesystem::path> paths_;
        std::size_t                        idx_;
        std::optional<Image<BGR>>          current_;

        iterator() : idx_(0) {}  // end sentinel

        explicit iterator(std::vector<std::filesystem::path> paths)
            : paths_(std::move(paths)), idx_(0) { load_current(); }

        void load_current() {
            while (idx_ < paths_.size()) {
                auto result = improc::io::imread<BGR>(paths_[idx_].string());
                if (result) { current_ = std::move(*result); return; }
                std::cerr << "views::from_dir: skipping " << paths_[idx_]
                          << ": imread failed\n";
                ++idx_;
            }
            current_ = std::nullopt;
        }

        Image<BGR> operator*() const { return *current_; }

        iterator& operator++() { ++idx_; load_current(); return *this; }

        bool operator==(const iterator& o) const {
            return !current_.has_value() && !o.current_.has_value();
        }
        bool operator!=(const iterator& o) const { return !(*this == o); }
    };

    /// Scans the directory for files matching the given extensions (case-insensitive),
    /// then returns an iterator that loads each image lazily on dereference.
    iterator begin() const {
        std::vector<std::filesystem::path> paths;
        for (const auto& entry : std::filesystem::directory_iterator{dir_}) {
            if (!entry.is_regular_file()) continue;
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
            if (std::find(exts_.begin(), exts_.end(), ext) != exts_.end())
                paths.push_back(entry.path());
        }
        std::sort(paths.begin(), paths.end());
        return iterator{std::move(paths)};
    }

    iterator end() const { return iterator{}; }

private:
    std::filesystem::path    dir_;
    std::vector<std::string> exts_;
};

// ── Factory ───────────────────────────────────────────────────────────────────

/// views::from_dir("dataset/", {".jpg", ".png"})
/// Throws FileNotFoundError if dir does not exist.
inline DirView from_dir(std::filesystem::path dir, std::vector<std::string> exts) {
    return DirView{std::move(dir), std::move(exts)};
}

// ── operator| overloads ───────────────────────────────────────────────────────

/// DirView | views::transform(op)  →  CollectionTransformView<DirView, Op>
template<typename Op>
auto operator|(DirView view, TransformAdapter<Op> adapter)
    -> CollectionTransformView<DirView, Op>
{
    return {std::move(view), std::move(adapter.op)};
}

/// DirView | views::filter(pred)  →  FilterView<DirView, Pred>
template<typename Pred>
auto operator|(DirView view, FilterAdapter<Pred> adapter)
    -> FilterView<DirView, Pred>
{
    return {std::move(view), std::move(adapter.pred)};
}

/// DirView | views::take(n)  →  TakeView<DirView>
inline auto operator|(DirView view, TakeAdapter a)
    -> TakeView<DirView>
{
    return {std::move(view), a.n};
}

/// DirView | views::drop(n)  →  DropView<DirView>
inline auto operator|(DirView view, DropAdapter a)
    -> DropView<DirView>
{
    return {std::move(view), a.n};
}

} // namespace improc::views
