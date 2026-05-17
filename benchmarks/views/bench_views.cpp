// benchmarks/views/bench_views.cpp
//
// Lazy views throughput vs eager equivalent.
// Tests three scenarios where lazy evaluation is beneficial:
//
//   1. take(kTake): lazy transforms only kTake of kBatch images;
//      eager transforms all kBatch then discards the rest.
//      Expected speedup: kBatch / kTake = 16×.
//
//   2. filter+transform: lazy applies op only to passing elements;
//      naive eager transforms all then filters.
//      Mixed batch: half large (ops are expensive), half 16×16 (filtered out).
//      Expected speedup: ~2× (half the ops skipped).
//
//   3. batch(8): chunked iteration overhead.
//
// Run: ./build/improc_benchmarks --benchmark_filter="views"

#include <benchmark/benchmark.h>
#include <algorithm>
#include <vector>
#include "improc/core/pipeline.hpp"
#include "improc/views/views.hpp"

using namespace improc::core;
namespace views = improc::views;

namespace {
constexpr int kBatch = 256;
constexpr int kTake  = 16;   // 6.25% of batch — lazy skips 94%

std::vector<Image<BGR>> make_batch(int h, int w) {
    std::vector<Image<BGR>> batch;
    batch.reserve(kBatch);
    for (int i = 0; i < kBatch; ++i)
        batch.emplace_back(cv::Mat(h, w, CV_8UC3));
    return batch;
}

// Half images at target size, half at 16×16.
// Filter pred keeps only large images → lazy skips 50% of transforms.
std::vector<Image<BGR>> make_mixed_batch(int h, int w) {
    std::vector<Image<BGR>> batch;
    batch.reserve(kBatch);
    for (int i = 0; i < kBatch / 2; ++i)
        batch.emplace_back(cv::Mat(h, w, CV_8UC3));
    for (int i = 0; i < kBatch / 2; ++i)
        batch.emplace_back(cv::Mat(16, 16, CV_8UC3));
    return batch;
}
} // namespace

// ── Scenario 1: take(kTake) — lazy skips kBatch-kTake transforms ─────────────

static void BM_views_take_lazy(benchmark::State& state) {
    auto batch = make_batch(state.range(0), state.range(1));
    auto op = GaussianBlur{}.kernel_size(5);
    for (auto _ : state) {
        auto result = batch
            | views::transform(op)
            | views::take(kTake)
            | views::to<std::vector<Image<BGR>>>();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_views_take_lazy)->Args({224, 224})->Args({640, 640});

static void BM_views_take_eager(benchmark::State& state) {
    auto batch = make_batch(state.range(0), state.range(1));
    auto op = GaussianBlur{}.kernel_size(5);
    for (auto _ : state) {
        std::vector<Image<BGR>> result;
        result.reserve(kBatch);
        for (const auto& img : batch)
            result.push_back(op(img));   // transforms all kBatch images
        result.erase(result.begin() + kTake, result.end()); // discard all but kTake
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_views_take_eager)->Args({224, 224})->Args({640, 640});

// ── Scenario 2: filter+transform — op runs only on passing elements ───────────

static void BM_views_filter_lazy(benchmark::State& state) {
    auto batch = make_mixed_batch(state.range(0), state.range(1));
    auto op    = GaussianBlur{}.kernel_size(5);
    auto pred  = [](const Image<BGR>& img) { return img.cols() > 32; };
    for (auto _ : state) {
        // lazy: filter before transform — 16×16 images are never processed
        auto result = batch
            | views::filter(pred)
            | views::transform(op)
            | views::to<std::vector<Image<BGR>>>();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_views_filter_lazy)->Args({224, 224})->Args({640, 640});

static void BM_views_filter_eager(benchmark::State& state) {
    auto batch = make_mixed_batch(state.range(0), state.range(1));
    auto op    = GaussianBlur{}.kernel_size(5);
    auto pred  = [](const Image<BGR>& img) { return img.cols() > 32; };
    for (auto _ : state) {
        // naive eager: transform all (including 16×16), then filter
        std::vector<Image<BGR>> all;
        all.reserve(batch.size());
        for (const auto& img : batch)
            all.push_back(op(img));
        std::vector<Image<BGR>> result;
        for (auto& img : all)
            if (pred(img))
                result.push_back(std::move(img));
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_views_filter_eager)->Args({224, 224})->Args({640, 640});

// ── Scenario 3: batch(8) chunking overhead ───────────────────────────────────

static void BM_views_batch(benchmark::State& state) {
    auto batch = make_batch(state.range(0), state.range(1));
    for (auto _ : state) {
        int count = 0;
        for (const auto& chunk : batch | views::batch(8))
            count += static_cast<int>(chunk.size());
        benchmark::DoNotOptimize(count);
    }
}
BENCHMARK(BM_views_batch)->Args({224, 224})->Args({640, 640});
