// benchmarks/views/bench_views.cpp
//
// Lazy views throughput vs eager std::transform equivalent.
// Collection of 32 pre-generated images; Resize as the representative op.
// Parametrized by image size.
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
constexpr int kBatchSize = 32;

std::vector<Image<BGR>> make_batch(int h, int w) {
    std::vector<Image<BGR>> batch;
    batch.reserve(kBatchSize);
    for (int i = 0; i < kBatchSize; ++i)
        batch.emplace_back(cv::Mat(h, w, CV_8UC3));
    return batch;
}
} // namespace

// ── views::transform (lazy) vs eager loop ────────────────────────────────────

static void BM_views_transform_lazy(benchmark::State& state) {
    auto batch = make_batch(state.range(0), state.range(1));
    auto op = Resize{}.width(64).height(64);
    for (auto _ : state) {
        auto result = batch
            | views::transform(op)
            | views::to<std::vector<Image<BGR>>>();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_views_transform_lazy)->Args({224, 224})->Args({640, 640});

static void BM_views_transform_eager(benchmark::State& state) {
    auto batch = make_batch(state.range(0), state.range(1));
    auto op = Resize{}.width(64).height(64);
    for (auto _ : state) {
        std::vector<Image<BGR>> result;
        result.reserve(batch.size());
        for (const auto& img : batch)
            result.push_back(op(img));
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_views_transform_eager)->Args({224, 224})->Args({640, 640});

// ── views::filter + transform chain vs eager loop ────────────────────────────

static void BM_views_filter_transform_lazy(benchmark::State& state) {
    auto batch = make_batch(state.range(0), state.range(1));
    auto op = Resize{}.width(64).height(64);
    auto pred = [](const Image<BGR>& img) { return img.cols() >= 64; };
    for (auto _ : state) {
        auto result = batch
            | views::filter(pred)
            | views::transform(op)
            | views::to<std::vector<Image<BGR>>>();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_views_filter_transform_lazy)->Args({224, 224})->Args({640, 640});

static void BM_views_filter_transform_eager(benchmark::State& state) {
    auto batch = make_batch(state.range(0), state.range(1));
    auto op = Resize{}.width(64).height(64);
    for (auto _ : state) {
        std::vector<Image<BGR>> result;
        for (const auto& img : batch)
            if (img.cols() >= 64)
                result.push_back(op(img));
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_views_filter_transform_eager)->Args({224, 224})->Args({640, 640});

// ── views::batch(8) throughput ───────────────────────────────────────────────

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
