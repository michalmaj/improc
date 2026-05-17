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

// ── AdaptiveThreshold (v0.2.0) ───────────────────────────────────────────────

static void BM_raw_adaptive_threshold(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC1);
    cv::Mat dst;
    for (auto _ : state) {
        cv::adaptiveThreshold(src, dst, 255,
            cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2.0);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_adaptive_threshold)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_adaptive_threshold(benchmark::State& state) {
    Image<Gray> img(cv::Mat(state.range(0), state.range(1), CV_8UC1));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | AdaptiveThreshold{}.block_size(11).C(2.0));
}
BENCHMARK(BM_improc_adaptive_threshold)->Args({480, 640})->Args({1080, 1920});

// ── HistogramEqualization (v0.2.0) ───────────────────────────────────────────

static void BM_raw_hist_eq(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC1);
    cv::Mat dst;
    for (auto _ : state) {
        cv::equalizeHist(src, dst);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_hist_eq)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_hist_eq(benchmark::State& state) {
    Image<Gray> img(cv::Mat(state.range(0), state.range(1), CV_8UC1));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | HistogramEqualization{});
}
BENCHMARK(BM_improc_hist_eq)->Args({480, 640})->Args({1080, 1920});

// ── NLMeansDenoising (v0.2.0) — slow by design, expected to dominate ─────────

static void BM_raw_nlmeans(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC3);
    cv::Mat dst;
    for (auto _ : state) {
        cv::fastNlMeansDenoisingColored(src, dst);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_nlmeans)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_nlmeans(benchmark::State& state) {
    Image<BGR> img(cv::Mat(state.range(0), state.range(1), CV_8UC3));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | NLMeansDenoising{});
}
BENCHMARK(BM_improc_nlmeans)->Args({480, 640})->Args({1080, 1920});

// ── MorphOpen (v0.2.0) ───────────────────────────────────────────────────────

static void BM_raw_morph_open(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC1);
    cv::Mat dst;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    for (auto _ : state) {
        cv::morphologyEx(src, dst, cv::MORPH_OPEN, kernel);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_morph_open)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_morph_open(benchmark::State& state) {
    Image<Gray> img(cv::Mat(state.range(0), state.range(1), CV_8UC1));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | MorphOpen{}.kernel_size(5));
}
BENCHMARK(BM_improc_morph_open)->Args({480, 640})->Args({1080, 1920});

// ── MorphClose (v0.2.0) ──────────────────────────────────────────────────────

static void BM_raw_morph_close(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC1);
    cv::Mat dst;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    for (auto _ : state) {
        cv::morphologyEx(src, dst, cv::MORPH_CLOSE, kernel);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_morph_close)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_morph_close(benchmark::State& state) {
    Image<Gray> img(cv::Mat(state.range(0), state.range(1), CV_8UC1));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | MorphClose{}.kernel_size(5));
}
BENCHMARK(BM_improc_morph_close)->Args({480, 640})->Args({1080, 1920});

// ── Invert (v0.2.0) ──────────────────────────────────────────────────────────

static void BM_raw_invert(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC3);
    cv::Mat dst;
    for (auto _ : state) {
        cv::bitwise_not(src, dst);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_invert)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_invert(benchmark::State& state) {
    Image<BGR> img(cv::Mat(state.range(0), state.range(1), CV_8UC3));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | Invert{});
}
BENCHMARK(BM_improc_invert)->Args({480, 640})->Args({1080, 1920});

// ── InRange (v0.2.0) ─────────────────────────────────────────────────────────

static void BM_raw_in_range(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC3);
    cv::Mat dst;
    for (auto _ : state) {
        cv::inRange(src, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), dst);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_in_range)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_in_range(benchmark::State& state) {
    Image<BGR> img(cv::Mat(state.range(0), state.range(1), CV_8UC3));
    for (auto _ : state)
        benchmark::DoNotOptimize(
            img | InRange{}.lower({0, 100, 100}).upper({10, 255, 255}));
}
BENCHMARK(BM_improc_in_range)->Args({480, 640})->Args({1080, 1920});

// ── MorphGradient (v0.3.0) ───────────────────────────────────────────────────

static void BM_raw_morph_gradient(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC1);
    cv::Mat dst;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    for (auto _ : state) {
        cv::morphologyEx(src, dst, cv::MORPH_GRADIENT, kernel);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_morph_gradient)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_morph_gradient(benchmark::State& state) {
    Image<Gray> img(cv::Mat(state.range(0), state.range(1), CV_8UC1));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | MorphGradient{}.kernel_size(5));
}
BENCHMARK(BM_improc_morph_gradient)->Args({480, 640})->Args({1080, 1920});

// ── TopHat (v0.3.0) ──────────────────────────────────────────────────────────

static void BM_raw_top_hat(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC1);
    cv::Mat dst;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    for (auto _ : state) {
        cv::morphologyEx(src, dst, cv::MORPH_TOPHAT, kernel);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_top_hat)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_top_hat(benchmark::State& state) {
    Image<Gray> img(cv::Mat(state.range(0), state.range(1), CV_8UC1));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | TopHat{}.kernel_size(5));
}
BENCHMARK(BM_improc_top_hat)->Args({480, 640})->Args({1080, 1920});

// ── BlackHat (v0.3.0) ────────────────────────────────────────────────────────

static void BM_raw_black_hat(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC1);
    cv::Mat dst;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    for (auto _ : state) {
        cv::morphologyEx(src, dst, cv::MORPH_BLACKHAT, kernel);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_black_hat)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_black_hat(benchmark::State& state) {
    Image<Gray> img(cv::Mat(state.range(0), state.range(1), CV_8UC1));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | BlackHat{}.kernel_size(5));
}
BENCHMARK(BM_improc_black_hat)->Args({480, 640})->Args({1080, 1920});

// ── HarrisCorner (v0.3.0) ────────────────────────────────────────────────────

static void BM_raw_harris(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC1);
    cv::Mat dst, dst_norm;
    for (auto _ : state) {
        cv::cornerHarris(src, dst, 2, 3, 0.04);
        cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_8U);  // HarrisCorner{} does both internally
        benchmark::DoNotOptimize(dst_norm);
    }
}
BENCHMARK(BM_raw_harris)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_harris(benchmark::State& state) {
    Image<Gray> img(cv::Mat(state.range(0), state.range(1), CV_8UC1));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | HarrisCorner{});
}
BENCHMARK(BM_improc_harris)->Args({480, 640})->Args({1080, 1920});

// ── ToLAB (v0.3.0) ───────────────────────────────────────────────────────────

static void BM_raw_to_lab(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC3);
    cv::Mat dst;
    for (auto _ : state) {
        cv::cvtColor(src, dst, cv::COLOR_BGR2Lab);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_to_lab)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_to_lab(benchmark::State& state) {
    Image<BGR> img(cv::Mat(state.range(0), state.range(1), CV_8UC3));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | ToLAB{});
}
BENCHMARK(BM_improc_to_lab)->Args({480, 640})->Args({1080, 1920});

// ── ToYCrCb (v0.3.0) ─────────────────────────────────────────────────────────

static void BM_raw_to_ycrcb(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC3);
    cv::Mat dst;
    for (auto _ : state) {
        cv::cvtColor(src, dst, cv::COLOR_BGR2YCrCb);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_to_ycrcb)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_to_ycrcb(benchmark::State& state) {
    Image<BGR> img(cv::Mat(state.range(0), state.range(1), CV_8UC3));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | ToYCrCb{});
}
BENCHMARK(BM_improc_to_ycrcb)->Args({480, 640})->Args({1080, 1920});

// ── PyrDown (v0.3.0) ─────────────────────────────────────────────────────────

static void BM_raw_pyr_down(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC3);
    cv::Mat dst;
    for (auto _ : state) {
        cv::pyrDown(src, dst);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_pyr_down)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_pyr_down(benchmark::State& state) {
    Image<BGR> img(cv::Mat(state.range(0), state.range(1), CV_8UC3));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | PyrDown{});
}
BENCHMARK(BM_improc_pyr_down)->Args({480, 640})->Args({1080, 1920});

// ── PyrUp (v0.3.0) ───────────────────────────────────────────────────────────

static void BM_raw_pyr_up(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC3);
    cv::Mat dst;
    for (auto _ : state) {
        cv::pyrUp(src, dst);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_pyr_up)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_pyr_up(benchmark::State& state) {
    Image<BGR> img(cv::Mat(state.range(0), state.range(1), CV_8UC3));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | PyrUp{});
}
BENCHMARK(BM_improc_pyr_up)->Args({480, 640})->Args({1080, 1920});

BENCHMARK_MAIN();
