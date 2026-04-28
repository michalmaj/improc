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
 * | Symbol                        | Description                                        |
 * |-------------------------------|----------------------------------------------------|
 * | `views::transform(op)`        | Applies op lazily to each element                  |
 * | `views::filter(pred)`         | Skips elements where pred returns false            |
 * | `views::take(n)`              | Stops after n elements                             |
 * | `views::drop(n)`              | Skips the first n elements                         |
 * | `views::to<vector<Image<F>>>()` | Collects all elements into a vector              |
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
 * // Collection — lazy pipeline
 * auto batch = images
 *     | views::transform(Resize{}.width(224).height(224))
 *     | views::filter([](const Image<BGR>& img) { return img.cols() > 0; })
 *     | views::drop(10)
 *     | views::take(32)
 *     | views::to<std::vector<Image<BGR>>>();
 * @endcode
 */
#include "improc/views/transform.hpp"
#include "improc/views/to.hpp"
#include "improc/views/collection.hpp"
#include "improc/views/filter.hpp"
#include "improc/views/take_drop.hpp"
