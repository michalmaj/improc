// include/improc/views/views.hpp
#pragma once

/**
 * @brief Umbrella include for `improc::views` — the lazy image pipeline namespace.
 *
 * Unlike the eager `operator|` in `improc::core`, adapters from `improc::views`
 * build a deferred computation that executes only when `views::to<T>()` is called.
 *
 * ### M1 — Single Image (available)
 *
 * | Symbol                    | Description                                     |
 * |---------------------------|-------------------------------------------------|
 * | `views::transform(op)`    | Returns a lazy adapter; no computation yet      |
 * | `views::to<Image<F>>()`   | Materializes the view; triggers the full chain  |
 *
 * ### Usage
 * @code
 * #include "improc/views/views.hpp"
 * namespace views = improc::views;
 *
 * auto view = img
 *     | views::transform(Resize{}.width(224).height(224))
 *     | views::transform(GaussianBlur{}.kernel_size(3));
 *
 * Image<BGR> result = view | views::to<Image<BGR>>();
 * @endcode
 */
#include "improc/views/transform.hpp"
#include "improc/views/to.hpp"
#include "improc/views/collection.hpp"
