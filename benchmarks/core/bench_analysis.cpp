// benchmarks/core/bench_analysis.cpp
//
// Image analysis throughput: FindContours, ConnectedComponents, DistanceTransform.
// All ops require binary Image<Gray> input — generated via threshold.
// Parametrized: ->Args({H, W}).
//
// Run: ./build/improc_benchmarks --benchmark_filter="contours|components|distance"

#include <benchmark/benchmark.h>
#include <opencv2/imgproc.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

namespace {
// Returns a binary cv::Mat with random blobs for realistic benchmark input.
cv::Mat make_binary(int h, int w) {
    cv::Mat src(h, w, CV_8UC1);
    cv::randu(src, cv::Scalar(0), cv::Scalar(255));
    cv::Mat binary;
    cv::threshold(src, binary, 128, 255, cv::THRESH_BINARY);
    return binary;
}
} // namespace

// ── FindContours ─────────────────────────────────────────────────────────────

static void BM_raw_find_contours(benchmark::State& state) {
    cv::Mat src = make_binary(state.range(0), state.range(1));
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    for (auto _ : state) {
        contours.clear();
        hierarchy.clear();
        cv::findContours(src, contours, hierarchy,
                         cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        benchmark::DoNotOptimize(contours);
    }
}
BENCHMARK(BM_raw_find_contours)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_find_contours(benchmark::State& state) {
    Image<Gray> img(make_binary(state.range(0), state.range(1)));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | FindContours{});
}
BENCHMARK(BM_improc_find_contours)->Args({480, 640})->Args({1080, 1920});

// ── ConnectedComponents ──────────────────────────────────────────────────────

static void BM_raw_connected_components(benchmark::State& state) {
    cv::Mat src = make_binary(state.range(0), state.range(1));
    cv::Mat labels, stats, centroids;
    for (auto _ : state) {
        cv::connectedComponentsWithStats(src, labels, stats, centroids);
        benchmark::DoNotOptimize(labels);
    }
}
BENCHMARK(BM_raw_connected_components)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_connected_components(benchmark::State& state) {
    Image<Gray> img(make_binary(state.range(0), state.range(1)));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | ConnectedComponents{});
}
BENCHMARK(BM_improc_connected_components)->Args({480, 640})->Args({1080, 1920});

// ── DistanceTransform ────────────────────────────────────────────────────────

static void BM_raw_distance_transform(benchmark::State& state) {
    cv::Mat src = make_binary(state.range(0), state.range(1));
    cv::Mat dst;
    for (auto _ : state) {
        cv::distanceTransform(src, dst, cv::DIST_L2, cv::DIST_MASK_3);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_distance_transform)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_distance_transform(benchmark::State& state) {
    Image<Gray> img(make_binary(state.range(0), state.range(1)));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | DistanceTransform{});
}
BENCHMARK(BM_improc_distance_transform)->Args({480, 640})->Args({1080, 1920});
