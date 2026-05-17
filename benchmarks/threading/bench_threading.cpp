// benchmarks/threading/bench_threading.cpp
//
// ThreadPool throughput: submit() latency and sequential vs threaded
// frame processing (Resize + GaussianBlur per frame).
// Parametrized by thread count.
//
// Run: ./build/improc_benchmarks --benchmark_filter="threading"

#include <benchmark/benchmark.h>
#include <thread>
#include <vector>
#include <future>
#include "improc/threading/thread_pool.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::threading;
using namespace improc::core;

namespace {
constexpr int kFrames = 16;

std::vector<Image<BGR>> make_frames(int h, int w) {
    std::vector<Image<BGR>> frames;
    frames.reserve(kFrames);
    for (int i = 0; i < kFrames; ++i)
        frames.emplace_back(cv::Mat(h, w, CV_8UC3));
    return frames;
}
} // namespace

// ── ThreadPool::submit() latency (trivial work) ───────────────────────────────

static void BM_thread_pool_submit(benchmark::State& state) {
    int n_threads = static_cast<int>(state.range(0));
    ThreadPool pool(n_threads);
    for (auto _ : state) {
        auto fut = pool.submit([]() { return 42; });
        benchmark::DoNotOptimize(fut.get());
    }
}
BENCHMARK(BM_thread_pool_submit)
    ->Arg(1)->Arg(2)->Arg(4)
    ->Arg(static_cast<int64_t>(std::thread::hardware_concurrency()));

// ── Sequential vs threaded frame processing ──────────────────────────────────

static void BM_sequential_frames(benchmark::State& state) {
    auto frames = make_frames(480, 640);
    for (auto _ : state) {
        for (const auto& f : frames) {
            auto result = f
                | Resize{}.width(224).height(224)
                | GaussianBlur{}.kernel_size(3);
            benchmark::DoNotOptimize(result);
        }
    }
}
BENCHMARK(BM_sequential_frames);

static void BM_threaded_frames(benchmark::State& state) {
    int n_threads = static_cast<int>(state.range(0));
    auto frames = make_frames(480, 640);
    ThreadPool pool(n_threads);
    for (auto _ : state) {
        std::vector<std::future<Image<BGR>>> futures;
        futures.reserve(frames.size());
        for (const auto& f : frames) {
            futures.push_back(pool.submit([f]() {
                return f
                    | Resize{}.width(224).height(224)
                    | GaussianBlur{}.kernel_size(3);
            }));
        }
        for (auto& fut : futures)
            benchmark::DoNotOptimize(fut.get());
    }
}
BENCHMARK(BM_threaded_frames)
    ->Arg(2)->Arg(4)
    ->Arg(static_cast<int64_t>(std::thread::hardware_concurrency()));
