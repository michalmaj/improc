// benchmarks/ml/bench_augmentation.cpp
//
// Augmentation throughput: single ops, Compose, MixUp, CutMix, BBoxCompose.
// No raw OpenCV comparison — measures absolute throughput.
// Parametrized by training image size: ->Args({224,224})->Args({640,640}).
//
// Run: ./build/improc_benchmarks --benchmark_filter="augment"

#include <benchmark/benchmark.h>
#include <random>
#include "improc/ml/augmentation.hpp"
#include "improc/ml/labeled.hpp"
#include "improc/ml/annotated.hpp"

using namespace improc::ml;
using namespace improc::core;

namespace {
std::mt19937 rng{42};

LabeledImage<BGR> make_labeled(int h, int w) {
    return { Image<BGR>(cv::Mat(h, w, CV_8UC3, cv::Scalar(128, 128, 128))),
             {1.f, 0.f, 0.f} };
}

AnnotatedImage<BGR> make_annotated(int h, int w) {
    BBox b;
    b.box      = cv::Rect2f(0.f, 0.f, w * 0.5f, h * 0.5f);
    b.class_id = 0;
    return { Image<BGR>(cv::Mat(h, w, CV_8UC3, cv::Scalar(128, 128, 128))), {b} };
}
} // namespace

// ── RandomFlip ───────────────────────────────────────────────────────────────

static void BM_augment_random_flip(benchmark::State& state) {
    Image<BGR> img(cv::Mat(state.range(0), state.range(1), CV_8UC3));
    RandomFlip op{};
    for (auto _ : state)
        benchmark::DoNotOptimize(op(img, rng));
}
BENCHMARK(BM_augment_random_flip)->Args({224, 224})->Args({640, 640});

// ── ColorJitter ──────────────────────────────────────────────────────────────

static void BM_augment_color_jitter(benchmark::State& state) {
    Image<BGR> img(cv::Mat(state.range(0), state.range(1), CV_8UC3));
    ColorJitter op{};
    op.brightness(0.7f, 1.3f).contrast(0.7f, 1.3f).saturation(0.7f, 1.3f);
    for (auto _ : state)
        benchmark::DoNotOptimize(op(img, rng));
}
BENCHMARK(BM_augment_color_jitter)->Args({224, 224})->Args({640, 640});

// ── RandomRotate ─────────────────────────────────────────────────────────────

static void BM_augment_random_rotate(benchmark::State& state) {
    Image<BGR> img(cv::Mat(state.range(0), state.range(1), CV_8UC3));
    RandomRotate op{};
    op.range(-30.0f, 30.0f);
    for (auto _ : state)
        benchmark::DoNotOptimize(op(img, rng));
}
BENCHMARK(BM_augment_random_rotate)->Args({224, 224})->Args({640, 640});

// ── RandomGaussianNoise ───────────────────────────────────────────────────────

static void BM_augment_gaussian_noise(benchmark::State& state) {
    Image<BGR> img(cv::Mat(state.range(0), state.range(1), CV_8UC3));
    RandomGaussianNoise op{};
    op.std_dev(5.0f, 15.0f);
    for (auto _ : state)
        benchmark::DoNotOptimize(op(img, rng));
}
BENCHMARK(BM_augment_gaussian_noise)->Args({224, 224})->Args({640, 640});

// ── Compose (flip + jitter + noise) ──────────────────────────────────────────

static void BM_augment_compose_3op(benchmark::State& state) {
    Image<BGR> img(cv::Mat(state.range(0), state.range(1), CV_8UC3));
    Compose<BGR> pipeline{};
    pipeline
        .add(RandomFlip{})
        .add(ColorJitter{}.brightness(0.7f, 1.3f).contrast(0.7f, 1.3f))
        .add(RandomGaussianNoise{}.std_dev(5.0f, 15.0f));
    for (auto _ : state)
        benchmark::DoNotOptimize(pipeline(img, rng));
}
BENCHMARK(BM_augment_compose_3op)->Args({224, 224})->Args({640, 640});

// ── RandomApply (p=0.5) ──────────────────────────────────────────────────────

static void BM_augment_random_apply(benchmark::State& state) {
    Image<BGR> img(cv::Mat(state.range(0), state.range(1), CV_8UC3));
    RandomApply<BGR> op{RandomFlip{}, 0.5f};
    for (auto _ : state)
        benchmark::DoNotOptimize(op(img, rng));
}
BENCHMARK(BM_augment_random_apply)->Args({224, 224})->Args({640, 640});

// ── MixUp ────────────────────────────────────────────────────────────────────

static void BM_augment_mixup(benchmark::State& state) {
    auto a = make_labeled(state.range(0), state.range(1));
    auto b = make_labeled(state.range(0), state.range(1));
    MixUp op{};
    op.alpha(0.4f);
    for (auto _ : state)
        benchmark::DoNotOptimize(op(a, b, rng));
}
BENCHMARK(BM_augment_mixup)->Args({224, 224})->Args({640, 640});

// ── CutMix ───────────────────────────────────────────────────────────────────

static void BM_augment_cutmix(benchmark::State& state) {
    auto a = make_labeled(state.range(0), state.range(1));
    auto b = make_labeled(state.range(0), state.range(1));
    CutMix op{};
    op.alpha(1.0f);
    for (auto _ : state)
        benchmark::DoNotOptimize(op(a, b, rng));
}
BENCHMARK(BM_augment_cutmix)->Args({224, 224})->Args({640, 640});

// ── BBoxCompose (flip + rotate) ───────────────────────────────────────────────

static void BM_augment_bbox_compose(benchmark::State& state) {
    auto sample = make_annotated(state.range(0), state.range(1));
    BBoxCompose<BGR> pipeline{};
    pipeline
        .add([](auto a, auto& r) { return RandomFlip{}.p(0.5f)(std::move(a), r); })
        .add([](auto a, auto& r) { return RandomRotate{}.range(-15.0f, 15.0f)(std::move(a), r); });
    for (auto _ : state)
        benchmark::DoNotOptimize(pipeline(sample, rng));
}
BENCHMARK(BM_augment_bbox_compose)->Args({224, 224})->Args({640, 640});
