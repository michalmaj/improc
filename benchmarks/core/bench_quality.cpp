// benchmarks/core/bench_quality.cpp
//
// Quality metrics + perceptual hash — throughput benchmarks at 480x640.
// Run: ./build/improc_benchmarks --benchmark_filter="psnr|ssim|gmsd|mse|hash"

#include <benchmark/benchmark.h>
#include <opencv2/imgproc.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

namespace {
Image<BGR> make_bgr(int h, int w) {
    cv::Mat m(h, w, CV_8UC3);
    cv::randu(m, 0, 255);
    return Image<BGR>(m);
}
} // namespace

// ── Quality metrics ───────────────────────────────────────────────────────────

static void BM_psnr(benchmark::State& state) {
    auto ref = make_bgr(480, 640);
    auto cmp = make_bgr(480, 640);
    for (auto _ : state)
        benchmark::DoNotOptimize(PSNR{}(ref, cmp));
}
BENCHMARK(BM_psnr);

static void BM_ssim(benchmark::State& state) {
    auto ref = make_bgr(480, 640);
    auto cmp = make_bgr(480, 640);
    for (auto _ : state)
        benchmark::DoNotOptimize(SSIM{}(ref, cmp));
}
BENCHMARK(BM_ssim);

static void BM_gmsd(benchmark::State& state) {
    auto ref = make_bgr(480, 640);
    auto cmp = make_bgr(480, 640);
    for (auto _ : state)
        benchmark::DoNotOptimize(GMSD{}(ref, cmp));
}
BENCHMARK(BM_gmsd);

static void BM_mse(benchmark::State& state) {
    auto ref = make_bgr(480, 640);
    auto cmp = make_bgr(480, 640);
    for (auto _ : state)
        benchmark::DoNotOptimize(MSE{}(ref, cmp));
}
BENCHMARK(BM_mse);

// ── Image hashing ─────────────────────────────────────────────────────────────

static void BM_average_hash(benchmark::State& state) {
    auto img = make_bgr(480, 640);
    for (auto _ : state)
        benchmark::DoNotOptimize(AverageHash{}(img));
}
BENCHMARK(BM_average_hash);

static void BM_phash(benchmark::State& state) {
    auto img = make_bgr(480, 640);
    for (auto _ : state)
        benchmark::DoNotOptimize(PHash{}(img));
}
BENCHMARK(BM_phash);

static void BM_marr_hildreth_hash(benchmark::State& state) {
    auto img = make_bgr(480, 640);
    for (auto _ : state)
        benchmark::DoNotOptimize(MarrHildrethHash{}(img));
}
BENCHMARK(BM_marr_hildreth_hash);

static void BM_radial_variance_hash(benchmark::State& state) {
    auto img = make_bgr(480, 640);
    for (auto _ : state)
        benchmark::DoNotOptimize(RadialVarianceHash{}(img));
}
BENCHMARK(BM_radial_variance_hash);

static void BM_color_moment_hash(benchmark::State& state) {
    auto img = make_bgr(480, 640);
    for (auto _ : state)
        benchmark::DoNotOptimize(ColorMomentHash{}(img));
}
BENCHMARK(BM_color_moment_hash);

static void BM_block_mean_hash(benchmark::State& state) {
    auto img = make_bgr(480, 640);
    for (auto _ : state)
        benchmark::DoNotOptimize(BlockMeanHash{}(img));
}
BENCHMARK(BM_block_mean_hash);

// ── Hash distance overhead (raw cv::norm vs wrapper) ─────────────────────────

static void BM_phash_distance(benchmark::State& state) {
    auto img = make_bgr(480, 640);
    auto h1 = PHash{}(img);
    auto h2 = PHash{}(img);
    for (auto _ : state)
        benchmark::DoNotOptimize(PHash::distance(h1, h2));
}
BENCHMARK(BM_phash_distance);

static void BM_radial_variance_distance(benchmark::State& state) {
    auto img = make_bgr(480, 640);
    auto h1 = RadialVarianceHash{}(img);
    auto h2 = RadialVarianceHash{}(img);
    for (auto _ : state)
        benchmark::DoNotOptimize(RadialVarianceHash::distance(h1, h2));
}
BENCHMARK(BM_radial_variance_distance);
