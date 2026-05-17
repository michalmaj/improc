// benchmarks/tracking/bench_tracking.cpp
//
// Per-frame tracker throughput: IouTracker, SortTracker, ByteTracker.
// All three receive identical synthetic detections for fair comparison.
// Parametrized by number of detections per frame.
//
// Run: ./build/improc_benchmarks --benchmark_filter="tracker"

#include <benchmark/benchmark.h>
#include "improc/ml/tracking/tracking.hpp"
#include "improc/ml/result_types.hpp"

using namespace improc::ml;

namespace {
std::vector<Detection> make_detections(int n) {
    std::vector<Detection> dets;
    dets.reserve(n);
    for (int i = 0; i < n; ++i) {
        Detection d;
        d.box        = cv::Rect2f(10.f * i, 10.f * i, 50.f, 50.f);
        d.class_id   = 0;
        d.confidence = 0.9f;
        dets.push_back(d);
    }
    return dets;
}
} // namespace

// ── IouTracker ───────────────────────────────────────────────────────────────

static void BM_iou_tracker_update(benchmark::State& state) {
    int n = static_cast<int>(state.range(0));
    auto dets = make_detections(n);
    IouTracker tracker{};
    for (auto _ : state) {
        benchmark::DoNotOptimize(tracker.update(dets));
    }
}
BENCHMARK(BM_iou_tracker_update)->Arg(10)->Arg(50)->Arg(100);

// ── SortTracker ──────────────────────────────────────────────────────────────

static void BM_sort_tracker_update(benchmark::State& state) {
    int n = static_cast<int>(state.range(0));
    auto dets = make_detections(n);
    SortTracker tracker{};
    for (auto _ : state) {
        benchmark::DoNotOptimize(tracker.update(dets));
    }
}
BENCHMARK(BM_sort_tracker_update)->Arg(10)->Arg(50)->Arg(100);

// ── ByteTracker ──────────────────────────────────────────────────────────────

static void BM_byte_tracker_update(benchmark::State& state) {
    int n = static_cast<int>(state.range(0));
    auto dets = make_detections(n);
    ByteTracker tracker{};
    for (auto _ : state) {
        benchmark::DoNotOptimize(tracker.update(dets));
    }
}
BENCHMARK(BM_byte_tracker_update)->Arg(10)->Arg(50)->Arg(100);
