// examples/views/demo_views_m1.cpp
// M1 demo: lazy single-image pipeline with views::transform / views::to<Image<F>>()
//
// Compares the eager operator| approach (executes immediately, one copy per op)
// with the lazy views:: approach (builds a deferred chain, single eval at the end).

#include <cassert>
#include <format>
#include <iostream>
#include "improc/core/pipeline.hpp"
#include "improc/views/views.hpp"

using namespace improc::core;
namespace views = improc::views;

int main() {
    // ── Setup: synthetic 640×480 BGR image ──────────────────────────────────
    cv::Mat mat(480, 640, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img{mat};

    std::cout << std::format("Source: {}×{}\n", img.cols(), img.rows());

    // ── Eager pipeline (existing improc::core operator|) ────────────────────
    // Each | executes immediately, producing an intermediate Image<BGR>.
    Image<BGR> eager = img
        | Resize{}.width(224).height(224)
        | GaussianBlur{}.kernel_size(5)
        | Brightness{}.delta(20.0);

    std::cout << std::format("Eager result:  {}×{}\n", eager.cols(), eager.rows());

    // ── Lazy pipeline (views::) ──────────────────────────────────────────────
    // Building the view does zero work — no pixels are touched yet.
    auto view = img
        | views::transform(Resize{}.width(224).height(224))
        | views::transform(GaussianBlur{}.kernel_size(5))
        | views::transform(Brightness{}.delta(20.0));

    // Evaluation only happens here, applying all three ops in one pass.
    Image<BGR> lazy = view | views::to<Image<BGR>>();

    std::cout << std::format("Lazy result:   {}×{}\n", lazy.cols(), lazy.rows());

    // ── Verify: source unchanged, results identical in size ──────────────────
    assert(img.cols() == 640 && img.rows() == 480);
    assert(eager.cols() == lazy.cols() && eager.rows() == lazy.rows());

    std::cout << "Source dimensions unchanged: OK\n";
    std::cout << "Eager == Lazy dimensions: OK\n";

    // ── Chaining multiple ops on the same source ─────────────────────────────
    auto thumbnail = img
        | views::transform(Resize{}.width(64).height(64))
        | views::transform(GaussianBlur{}.kernel_size(3))
        | views::to<Image<BGR>>();

    std::cout << std::format("Thumbnail: {}×{}\n", thumbnail.cols(), thumbnail.rows());

    return 0;
}
