// include/improc/views/views.hpp
#pragma once

/**
 * @brief Umbrella include for `improc::views` — the lazy image pipeline namespace.
 *
 * Unlike the eager `operator|` in `improc::core`, adapters from `improc::views`
 * build a deferred computation that executes only when `views::to<T>()` is called.
 *
 * ### M1 — Single Image
 *
 * | Symbol                    | Description                                     |
 * |---------------------------|-------------------------------------------------|
 * | `views::transform(op)`    | Returns a lazy adapter; no computation yet      |
 * | `views::to<Image<F>>()`   | Materializes the view; triggers the full chain  |
 *
 * ### M2 — In-Memory Collections (`std::vector<Image<F>>`)
 *
 * | Symbol                          | Description                                       |
 * |---------------------------------|---------------------------------------------------|
 * | `views::transform(op)`          | Applies op lazily to each element                 |
 * | `views::filter(pred)`           | Skips elements where pred returns false           |
 * | `views::take(n)`                | Stops after n elements                            |
 * | `views::drop(n)`                | Skips the first n elements                        |
 * | `views::to<vector<Image<F>>>()` | Collects all elements into a vector               |
 *
 * ### M3 — External Sources
 *
 * | Symbol                    | Description                                               |
 * |---------------------------|-----------------------------------------------------------|
 * | `views::VideoView`        | Lazy single-pass view over `improc::io::VideoReader`      |
 * | `views::from_dir(p, ext)` | Lazy view over image files in a directory by extension    |
 *
 * ### Usage
 * @code
 * #include "improc/views/views.hpp"
 * namespace views = improc::views;
 *
 * // Single image — lazy chain
 * auto view = img
 *     | views::transform(Resize{}.width(224).height(224))
 *     | views::transform(GaussianBlur{}.kernel_size(3));
 * Image<BGR> result = view | views::to<Image<BGR>>();
 *
 * // In-memory collection
 * auto batch = images
 *     | views::transform(Resize{}.width(224).height(224))
 *     | views::filter([](const Image<BGR>& img) { return img.cols() > 0; })
 *     | views::drop(10)
 *     | views::take(32)
 *     | views::to<std::vector<Image<BGR>>>();
 *
 * // VideoReader source — O(1) RAM regardless of video length
 * improc::io::VideoReader reader{"clip.mp4"};
 * for (const auto& frame : views::VideoView{reader} | views::take(100)) { ... }
 *
 * // Directory source — loads images lazily one at a time
 * auto dataset = views::from_dir("train/", {".jpg", ".png"})
 *     | views::transform(Resize{}.width(224).height(224))
 *     | views::to<std::vector<Image<BGR>>>();
 * @endcode
 */
#include "improc/views/transform.hpp"
#include "improc/views/to.hpp"
#include "improc/views/collection.hpp"
#include "improc/views/filter.hpp"
#include "improc/views/take_drop.hpp"
#include "improc/views/video_source.hpp"
#include "improc/views/from_dir.hpp"
