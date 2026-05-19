// benchmarks/core/bench_background_subtract.cpp
//
// Throughput: BackgroundSubtractMOG2 and BackgroundSubtractKNN on BGR frames.
// Stateful ops — model is warm from prior iterations (intentional).
// Parametrized: ->Args({H, W}).
//
// Run: ./build/improc_benchmarks --benchmark_filter="background_subtract"

#include <benchmark/benchmark.h>
#include <opencv2/core.hpp>
#include "improc/core/ops/background_subtract.hpp"
#include "improc/core/image.hpp"

using namespace improc::core;

namespace {
cv::Mat make_bgr_frame(int h, int w) {
    cv::Mat m(h, w, CV_8UC3);
    cv::randu(m, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    return m;
}
} // namespace

// ── BackgroundSubtractMOG2 ───────────────────────────────────────────────────

static void BM_raw_background_subtract_mog2(benchmark::State& state) {
    cv::Mat frame = make_bgr_frame(state.range(0), state.range(1));
    cv::Mat fg;
    auto sub = cv::createBackgroundSubtractorMOG2();
    for (auto _ : state) {
        sub->apply(frame, fg);
        benchmark::DoNotOptimize(fg);
    }
}
BENCHMARK(BM_raw_background_subtract_mog2)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_background_subtract_mog2(benchmark::State& state) {
    Image<BGR> frame(make_bgr_frame(state.range(0), state.range(1)));
    BackgroundSubtractMOG2 sub;
    for (auto _ : state) {
        benchmark::DoNotOptimize(sub(frame));
    }
}
BENCHMARK(BM_improc_background_subtract_mog2)->Args({480, 640})->Args({1080, 1920});

// ── BackgroundSubtractKNN ─────────────────────────────────────────────────────

static void BM_raw_background_subtract_knn(benchmark::State& state) {
    cv::Mat frame = make_bgr_frame(state.range(0), state.range(1));
    cv::Mat fg;
    auto sub = cv::createBackgroundSubtractorKNN();
    for (auto _ : state) {
        sub->apply(frame, fg);
        benchmark::DoNotOptimize(fg);
    }
}
BENCHMARK(BM_raw_background_subtract_knn)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_background_subtract_knn(benchmark::State& state) {
    Image<BGR> frame(make_bgr_frame(state.range(0), state.range(1)));
    BackgroundSubtractKNN sub;
    for (auto _ : state) {
        benchmark::DoNotOptimize(sub(frame));
    }
}
BENCHMARK(BM_improc_background_subtract_knn)->Args({480, 640})->Args({1080, 1920});
