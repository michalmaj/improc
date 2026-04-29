// examples/views/demo_views_m2.cpp
// M2 demo: lazy collection pipeline over std::vector<Image<BGR>>
// Shows transform, filter, take, drop and their compositions.

#include <format>
#include <iostream>
#include <vector>
#include "improc/core/pipeline.hpp"
#include "improc/views/views.hpp"

using namespace improc::core;
namespace views = improc::views;

static Image<BGR> make_image(int w, int h, int blue) {
    cv::Mat mat(h, w, CV_8UC3, cv::Scalar(blue, 100, 200));
    return Image<BGR>{mat};
}

int main() {
    // ── Build a batch of 10 images of varying sizes ──────────────────────────
    std::vector<Image<BGR>> batch;
    for (int i = 0; i < 10; ++i) {
        int side = (i % 2 == 0) ? 128 : 64;  // alternating 128px and 64px
        batch.push_back(make_image(side, side, i * 20));
    }
    std::cout << std::format("Source batch: {} images\n", batch.size());

    // ── transform: resize every image lazily ────────────────────────────────
    auto resized = batch
        | views::transform(Resize{}.width(32).height(32))
        | views::to<std::vector<Image<BGR>>>();

    std::cout << std::format("After transform(Resize→32×32): {} images, each {}×{}\n",
        resized.size(), resized[0].cols(), resized[0].rows());

    // ── filter: keep only the large (128px) images ──────────────────────────
    auto large_only = batch
        | views::filter([](const Image<BGR>& img) { return img.cols() == 128; })
        | views::to<std::vector<Image<BGR>>>();

    std::cout << std::format("After filter(cols==128): {} images\n", large_only.size());

    // ── take: first 3 images from batch ─────────────────────────────────────
    auto first_three = batch
        | views::take(3)
        | views::to<std::vector<Image<BGR>>>();

    std::cout << std::format("After take(3): {} images\n", first_three.size());

    // ── drop: skip the first 4, keep the rest ───────────────────────────────
    auto after_drop = batch
        | views::drop(4)
        | views::to<std::vector<Image<BGR>>>();

    std::cout << std::format("After drop(4): {} images\n", after_drop.size());

    // ── composition: filter → transform → take ───────────────────────────────
    // Keep large images, resize them, take the first 3.
    auto pipeline_result = batch
        | views::filter([](const Image<BGR>& img) { return img.cols() == 128; })
        | views::transform(Resize{}.width(64).height(64))
        | views::take(3)
        | views::to<std::vector<Image<BGR>>>();

    std::cout << std::format("filter(large) | transform(64×64) | take(3): {} images, each {}×{}\n",
        pipeline_result.size(), pipeline_result[0].cols(), pipeline_result[0].rows());

    // ── lazy range-for: no materialization ──────────────────────────────────
    // Process one image at a time — only the current image is in RAM.
    int processed = 0;
    for (const auto& img : batch | views::transform(Resize{}.width(16).height(16))) {
        (void)img;
        ++processed;
    }
    std::cout << std::format("Lazy range-for processed: {} images\n", processed);

    // ── source unchanged ─────────────────────────────────────────────────────
    std::cout << std::format("Source batch still has {} images\n", batch.size());

    return 0;
}
