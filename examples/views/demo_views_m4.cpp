// examples/views/demo_views_m4.cpp
// M4 demo: advanced adapters — batch, enumerate, zip
//
// batch:     group images into mini-batches for ML training loops
// enumerate: track the index of each element without losing laziness
// zip:       pair two sources element-by-element

#include <format>
#include <iostream>
#include <vector>
#include "improc/core/pipeline.hpp"
#include "improc/views/views.hpp"

using namespace improc::core;
namespace views = improc::views;

static Image<BGR> make_image(int w, int h, int blue) {
    return Image<BGR>{cv::Mat(h, w, CV_8UC3, cv::Scalar(blue, 100, 200))};
}

int main() {
    // Source: 10 images of two sizes
    std::vector<Image<BGR>> images;
    for (int i = 0; i < 10; ++i) {
        int side = (i < 5) ? 128 : 64;
        images.push_back(make_image(side, side, i * 20));
    }

    // ── batch: ML-style mini-batches ────────────────────────────────────────
    std::cout << "── batch(3) over 10 images:\n";
    int batch_no = 0;
    for (const auto& batch : images | views::batch(3)) {
        std::cout << std::format("  batch {}: {} images\n", batch_no++, batch.size());
    }

    // batch after transform
    std::cout << "\n── transform(32×32) | batch(4):\n";
    for (const auto& batch : images
            | views::transform(Resize{}.width(32).height(32))
            | views::batch(4)) {
        std::cout << std::format("  batch: {} images, each {}×{}\n",
            batch.size(), batch[0].cols(), batch[0].rows());
    }

    // ── enumerate: index tracking ────────────────────────────────────────────
    std::cout << "\n── enumerate over first 5 images:\n";
    for (const auto& [idx, img] : images | views::take(5) | views::enumerate) {
        std::cout << std::format("  [{}] {}×{}\n", idx, img.cols(), img.rows());
    }

    // enumerate after filter (indices reset to 0 for the filtered subsequence)
    std::cout << "\n── filter(large) | enumerate:\n";
    for (const auto& [idx, img] : images
            | views::filter([](const Image<BGR>& img) { return img.cols() == 128; })
            | views::enumerate) {
        std::cout << std::format("  [{}] {}×{}\n", idx, img.cols(), img.rows());
    }

    // ── zip: pair images with masks ──────────────────────────────────────────
    std::vector<Image<BGR>> masks;
    for (int i = 0; i < 7; ++i)
        masks.push_back(make_image(64, 64, 255 - i * 30));

    std::cout << "\n── zip(images[10], masks[7]) — stops at shorter (7):\n";
    std::size_t pair_count = 0;
    for (const auto& [img, mask] : views::zip(images, masks))
        ++pair_count;
    std::cout << std::format("  {} pairs produced\n", pair_count);

    // zip with a transformed source
    auto resized = images
        | views::transform(Resize{}.width(32).height(32))
        | views::to<std::vector<Image<BGR>>>();

    std::cout << "\n── zip(resized[10], masks[7]):\n";
    std::size_t zip_count = 0;
    for (const auto& [img, mask] : views::zip(resized, masks)) {
        (void)img; (void)mask;
        ++zip_count;
    }
    std::cout << std::format("  {} pairs (img 32×32, mask 64×64)\n", zip_count);

    std::cout << "\nDone.\n";
    return 0;
}
