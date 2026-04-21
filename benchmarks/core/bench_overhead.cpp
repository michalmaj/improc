// benchmarks/core/bench_overhead.cpp
//
// Zero-overhead suite: raw OpenCV vs improc++ per op on 480x640 BGR.
// Acceptance criterion: wrapper overhead <= 5ns per op.
//
// Build: cmake -DIMPROC_BENCHMARKS=ON ...
// Run:   ./build/improc_benchmarks --benchmark_filter="BM_raw|BM_improc"

#include <benchmark/benchmark.h>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

// ── Resize ────────────────────────────────────────────────────────────────────

static void BM_raw_resize(benchmark::State& state) {
    cv::Mat src(480, 640, CV_8UC3);
    cv::Mat dst;
    for (auto _ : state) {
        cv::resize(src, dst, cv::Size(224, 224));
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_resize);

static void BM_improc_resize(benchmark::State& state) {
    Image<BGR> img(cv::Mat(480, 640, CV_8UC3));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | Resize{}.width(224).height(224));
}
BENCHMARK(BM_improc_resize);

// ── GaussianBlur ──────────────────────────────────────────────────────────────

static void BM_raw_gaussian(benchmark::State& state) {
    cv::Mat src(480, 640, CV_8UC3);
    cv::Mat dst;
    for (auto _ : state) {
        cv::GaussianBlur(src, dst, cv::Size(3, 3), 0);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_gaussian);

static void BM_improc_gaussian(benchmark::State& state) {
    Image<BGR> img(cv::Mat(480, 640, CV_8UC3));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | GaussianBlur{}.kernel_size(3));
}
BENCHMARK(BM_improc_gaussian);

// ── GammaCorrection (8-bit LUT path) ─────────────────────────────────────────

static void BM_raw_gamma(benchmark::State& state) {
    cv::Mat src(480, 640, CV_8UC3);
    cv::Mat lut(1, 256, CV_8U);
    auto* p = lut.ptr<uchar>();
    for (int i = 0; i < 256; ++i)
        p[i] = cv::saturate_cast<uchar>(255.0 * std::pow(i / 255.0, 0.5));
    cv::Mat dst;
    for (auto _ : state) {
        cv::LUT(src, lut, dst);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_gamma);

static void BM_improc_gamma(benchmark::State& state) {
    Image<BGR> img(cv::Mat(480, 640, CV_8UC3));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | GammaCorrection{}.gamma(0.5f));
}
BENCHMARK(BM_improc_gamma);

// ── CLAHE (Gray) ──────────────────────────────────────────────────────────────

static void BM_raw_clahe(benchmark::State& state) {
    cv::Mat src(480, 640, CV_8UC1);
    cv::Mat dst;
    auto clahe = cv::createCLAHE(40.0, cv::Size(8, 8));
    for (auto _ : state) {
        clahe->apply(src, dst);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_clahe);

static void BM_improc_clahe(benchmark::State& state) {
    Image<Gray> img(cv::Mat(480, 640, CV_8UC1));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | CLAHE{});
}
BENCHMARK(BM_improc_clahe);

// ── BilateralFilter ───────────────────────────────────────────────────────────

static void BM_raw_bilateral(benchmark::State& state) {
    cv::Mat src(480, 640, CV_8UC3);
    cv::Mat dst;
    for (auto _ : state) {
        cv::bilateralFilter(src, dst, 9, 75.0, 75.0);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_bilateral);

static void BM_improc_bilateral(benchmark::State& state) {
    Image<BGR> img(cv::Mat(480, 640, CV_8UC3));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | BilateralFilter{});
}
BENCHMARK(BM_improc_bilateral);

// ── ToFloat32C3 ───────────────────────────────────────────────────────────────

static void BM_raw_to_float32c3(benchmark::State& state) {
    cv::Mat src(480, 640, CV_8UC3);
    cv::Mat dst;
    for (auto _ : state) {
        src.convertTo(dst, CV_32FC3, 1.0 / 255.0);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_to_float32c3);

static void BM_improc_to_float32c3(benchmark::State& state) {
    Image<BGR> img(cv::Mat(480, 640, CV_8UC3));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | ToFloat32C3{});
}
BENCHMARK(BM_improc_to_float32c3);

// ── NormalizeTo ───────────────────────────────────────────────────────────────

static void BM_raw_normalize_to(benchmark::State& state) {
    cv::Mat src(480, 640, CV_32FC3);
    cv::Mat dst;
    for (auto _ : state) {
        cv::normalize(src, dst, 0.0, 1.0, cv::NORM_MINMAX);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_normalize_to);

static void BM_improc_normalize_to(benchmark::State& state) {
    Image<Float32C3> img(cv::Mat(480, 640, CV_32FC3));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | NormalizeTo{0.0f, 1.0f});
}
BENCHMARK(BM_improc_normalize_to);

// ── WarpAffine ────────────────────────────────────────────────────────────────

static void BM_raw_warp_affine(benchmark::State& state) {
    cv::Mat src(480, 640, CV_8UC3);
    cv::Mat M = cv::Mat::eye(2, 3, CV_64F);
    cv::Mat dst;
    for (auto _ : state) {
        cv::warpAffine(src, dst, M, cv::Size(640, 480));
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_warp_affine);

static void BM_improc_warp_affine(benchmark::State& state) {
    Image<BGR> img(cv::Mat(480, 640, CV_8UC3));
    cv::Mat M = cv::Mat::eye(2, 3, CV_64F);
    for (auto _ : state)
        benchmark::DoNotOptimize(img | WarpAffine{}.matrix(M));
}
BENCHMARK(BM_improc_warp_affine);
