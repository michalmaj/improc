// benchmarks/core/bench_photo.cpp
//
// Photo + stitching ops — throughput benchmarks at 480x640.
// Build: cmake -DIMPROC_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
//              -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="conan_provider.cmake" \
//              -DCONAN_COMMAND=/opt/anaconda3/envs/conan_cv/bin/conan -B build .
//        cmake --build build --target improc_benchmarks --parallel 2
// Run:   ./build/improc_benchmarks --benchmark_filter="photo|stitch"

#include <benchmark/benchmark.h>
#include <opencv2/imgproc.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

namespace {

Image<BGR> make_bgr(int h, int w) {
    cv::Mat m(h, w, CV_8UC3);
    cv::randu(m, 0, 255);
    cv::GaussianBlur(m, m, {5, 5}, 1.5);
    return Image<BGR>(m);
}

Image<Float32C3> make_hdr(int h, int w) {
    cv::Mat m(h, w, CV_32FC3);
    cv::randu(m, 0.f, 1.f);
    return Image<Float32C3>(m);
}

Image<Gray> make_circle_mask(int h, int w) {
    cv::Mat m(h, w, CV_8UC1, cv::Scalar(0));
    cv::circle(m, {w / 2, h / 2}, h / 3, cv::Scalar(255), -1);
    return Image<Gray>(m);
}

} // namespace

static void BM_edge_preserving_filter(benchmark::State& state) {
    auto img = make_bgr(480, 640);
    for (auto _ : state)
        benchmark::DoNotOptimize(EdgePreservingFilter{}(img));
}
BENCHMARK(BM_edge_preserving_filter);

static void BM_detail_enhance(benchmark::State& state) {
    auto img = make_bgr(480, 640);
    for (auto _ : state)
        benchmark::DoNotOptimize(DetailEnhance{}(img));
}
BENCHMARK(BM_detail_enhance);

static void BM_stylize(benchmark::State& state) {
    auto img = make_bgr(480, 640);
    for (auto _ : state)
        benchmark::DoNotOptimize(Stylize{}(img));
}
BENCHMARK(BM_stylize);

static void BM_pencil_sketch(benchmark::State& state) {
    auto img = make_bgr(480, 640);
    for (auto _ : state)
        benchmark::DoNotOptimize(PencilSketch{}(img));
}
BENCHMARK(BM_pencil_sketch);

static void BM_seamless_clone(benchmark::State& state) {
    auto src  = make_bgr(480, 640);
    auto dst  = make_bgr(480, 640);
    auto mask = make_circle_mask(480, 640);
    for (auto _ : state)
        benchmark::DoNotOptimize(SeamlessClone{}(src, dst, mask, {320, 240}));
}
BENCHMARK(BM_seamless_clone);

static void BM_merge_hdr_mertens(benchmark::State& state) {
    std::vector<Image<BGR>> imgs{make_bgr(480, 640), make_bgr(480, 640), make_bgr(480, 640)};
    for (auto _ : state)
        benchmark::DoNotOptimize(MergeHDR{}.method(MergeHDR::Method::Mertens)(imgs));
}
BENCHMARK(BM_merge_hdr_mertens)->Iterations(5);

static void BM_tonemap(benchmark::State& state) {
    auto img = make_hdr(480, 640);
    for (auto _ : state)
        benchmark::DoNotOptimize(ToneMap{}(img));
}
BENCHMARK(BM_tonemap);

static void BM_stitch(benchmark::State& state) {
    cv::Mat base(480, 960, CV_8UC3);
    cv::randu(base, 50, 200);
    cv::GaussianBlur(base, base, {15, 15}, 3.0);
    Image<BGR> left(base(cv::Rect(0,   0, 720, 480)).clone());
    Image<BGR> right(base(cv::Rect(240, 0, 720, 480)).clone());
    for (auto _ : state)
        benchmark::DoNotOptimize(Stitch{}({left, right}));
}
BENCHMARK(BM_stitch)->Iterations(5);
