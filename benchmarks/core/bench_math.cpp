// benchmarks/core/bench_math.cpp
//
// Math & foundation ops — overhead suite: raw OpenCV vs improc++ wrapper.
// Acceptance criterion: wrapper overhead <= 5 ns per op.
//
// Build: cmake -DIMPROC_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
//              -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="conan_provider.cmake" -B build .
//        cmake --build build --target improc_benchmarks
// Run:   ./build/improc_benchmarks \
//          --benchmark_filter="add|subtract|multiply|divide|boxfilter|convolve|sobel|scharr|convert_scale|split|merge|integral|minmax|mean_stddev|nonzero|reduce"

#include <benchmark/benchmark.h>
#include <opencv2/imgproc.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

// ── Add ───────────────────────────────────────────────────────────────────────

static void BM_raw_add(benchmark::State& state) {
    cv::Mat a(state.range(0), state.range(1), CV_8UC3);
    cv::Mat b(state.range(0), state.range(1), CV_8UC3);
    for (auto _ : state) {
        cv::Mat dst;
        cv::add(a, b, dst);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_add)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_add(benchmark::State& state) {
    Image<BGR> img(cv::Mat(state.range(0), state.range(1), CV_8UC3));
    cv::Mat other(state.range(0), state.range(1), CV_8UC3);
    Add op(other);
    for (auto _ : state)
        benchmark::DoNotOptimize(op(img));
}
BENCHMARK(BM_improc_add)->Args({480, 640})->Args({1080, 1920});

// ── Subtract ─────────────────────────────────────────────────────────────────

static void BM_raw_subtract(benchmark::State& state) {
    cv::Mat a(state.range(0), state.range(1), CV_8UC3);
    cv::Mat b(state.range(0), state.range(1), CV_8UC3);
    for (auto _ : state) {
        cv::Mat dst;
        cv::subtract(a, b, dst);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_subtract)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_subtract(benchmark::State& state) {
    Image<BGR> img(cv::Mat(state.range(0), state.range(1), CV_8UC3));
    cv::Mat other(state.range(0), state.range(1), CV_8UC3);
    Subtract op(other);
    for (auto _ : state)
        benchmark::DoNotOptimize(op(img));
}
BENCHMARK(BM_improc_subtract)->Args({480, 640})->Args({1080, 1920});

// ── Multiply ─────────────────────────────────────────────────────────────────

static void BM_raw_multiply(benchmark::State& state) {
    cv::Mat a(state.range(0), state.range(1), CV_8UC3);
    cv::Mat b(state.range(0), state.range(1), CV_8UC3);
    for (auto _ : state) {
        cv::Mat dst;
        cv::multiply(a, b, dst);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_multiply)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_multiply(benchmark::State& state) {
    Image<BGR> img(cv::Mat(state.range(0), state.range(1), CV_8UC3));
    cv::Mat other(state.range(0), state.range(1), CV_8UC3);
    Multiply op(other);
    for (auto _ : state)
        benchmark::DoNotOptimize(op(img));
}
BENCHMARK(BM_improc_multiply)->Args({480, 640})->Args({1080, 1920});

// ── Divide ───────────────────────────────────────────────────────────────────

static void BM_raw_divide(benchmark::State& state) {
    cv::Mat a(state.range(0), state.range(1), CV_8UC3);
    cv::Mat b(state.range(0), state.range(1), CV_8UC3,
              cv::Scalar(128, 128, 128));
    for (auto _ : state) {
        cv::Mat dst;
        cv::divide(a, b, dst);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_divide)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_divide(benchmark::State& state) {
    Image<BGR> img(cv::Mat(state.range(0), state.range(1), CV_8UC3));
    cv::Mat other(state.range(0), state.range(1), CV_8UC3,
                  cv::Scalar(128, 128, 128));
    Divide op(other);
    for (auto _ : state)
        benchmark::DoNotOptimize(op(img));
}
BENCHMARK(BM_improc_divide)->Args({480, 640})->Args({1080, 1920});

// ── BoxFilter ─────────────────────────────────────────────────────────────────

static void BM_raw_boxfilter(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC3);
    for (auto _ : state) {
        cv::Mat dst;
        cv::boxFilter(src, dst, -1, cv::Size(5, 5));
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_boxfilter)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_boxfilter(benchmark::State& state) {
    Image<BGR> img(cv::Mat(state.range(0), state.range(1), CV_8UC3));
    for (auto _ : state)
        benchmark::DoNotOptimize(img | BoxFilter{}.kernel_size(5));
}
BENCHMARK(BM_improc_boxfilter)->Args({480, 640})->Args({1080, 1920});

// ── Convolve ─────────────────────────────────────────────────────────────────

static void BM_raw_convolve(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC3);
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_32F) / 9.0f;
    for (auto _ : state) {
        cv::Mat dst;
        cv::filter2D(src, dst, -1, kernel);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_convolve)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_convolve(benchmark::State& state) {
    Image<BGR> img(cv::Mat(state.range(0), state.range(1), CV_8UC3));
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_32F) / 9.0f;
    Convolve op(kernel);
    for (auto _ : state)
        benchmark::DoNotOptimize(op(img));
}
BENCHMARK(BM_improc_convolve)->Args({480, 640})->Args({1080, 1920});

// ── SobelGradient (480×640 Gray only) ────────────────────────────────────────

static void BM_raw_sobel_gradient(benchmark::State& state) {
    cv::Mat src(480, 640, CV_8UC1);
    for (auto _ : state) {
        cv::Mat dx, dy;
        cv::Sobel(src, dx, CV_16S, 1, 0);
        cv::Sobel(src, dy, CV_16S, 0, 1);
        benchmark::DoNotOptimize(dx);
        benchmark::DoNotOptimize(dy);
    }
}
BENCHMARK(BM_raw_sobel_gradient);

static void BM_improc_sobel_gradient(benchmark::State& state) {
    Image<Gray> img(cv::Mat(480, 640, CV_8UC1));
    SobelGradient op;
    for (auto _ : state)
        benchmark::DoNotOptimize(op(img));
}
BENCHMARK(BM_improc_sobel_gradient);

// ── ScharrGradient ───────────────────────────────────────────────────────────

static void BM_raw_scharr_gradient(benchmark::State& state) {
    cv::Mat src(480, 640, CV_8UC1);
    for (auto _ : state) {
        cv::Mat dx, dy;
        cv::Scharr(src, dx, CV_16S, 1, 0);
        cv::Scharr(src, dy, CV_16S, 0, 1);
        benchmark::DoNotOptimize(dx);
        benchmark::DoNotOptimize(dy);
    }
}
BENCHMARK(BM_raw_scharr_gradient);

static void BM_improc_scharr_gradient(benchmark::State& state) {
    Image<Gray> img(cv::Mat(480, 640, CV_8UC1));
    ScharrGradient op;
    for (auto _ : state)
        benchmark::DoNotOptimize(op(img));
}
BENCHMARK(BM_improc_scharr_gradient);

// ── ConvertScaleAbs — input is CV_16S Sobel output ───────────────────────────

static void BM_raw_convert_scale_abs(benchmark::State& state) {
    cv::Mat src(480, 640, CV_8UC1);
    cv::Mat dx;
    cv::Sobel(src, dx, CV_16S, 1, 0);  // pre-computed outside loop
    for (auto _ : state) {
        cv::Mat dst;
        cv::convertScaleAbs(dx, dst);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_convert_scale_abs);

static void BM_improc_convert_scale_abs(benchmark::State& state) {
    cv::Mat src(480, 640, CV_8UC1);
    cv::Mat dx;
    cv::Sobel(src, dx, CV_16S, 1, 0);  // pre-computed outside loop
    ConvertScaleAbs op;
    for (auto _ : state)
        benchmark::DoNotOptimize(op(dx));
}
BENCHMARK(BM_improc_convert_scale_abs);

// ── SplitChannels ─────────────────────────────────────────────────────────────

static void BM_raw_split_channels(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC3);
    for (auto _ : state) {
        std::vector<cv::Mat> channels;
        cv::split(src, channels);
        benchmark::DoNotOptimize(channels);
    }
}
BENCHMARK(BM_raw_split_channels)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_split_channels(benchmark::State& state) {
    Image<BGR> img(cv::Mat(state.range(0), state.range(1), CV_8UC3));
    SplitChannels op;
    for (auto _ : state)
        benchmark::DoNotOptimize(op(img));
}
BENCHMARK(BM_improc_split_channels)->Args({480, 640})->Args({1080, 1920});

// ── MergeChannels ─────────────────────────────────────────────────────────────

static void BM_raw_merge_channels(benchmark::State& state) {
    cv::Mat b(state.range(0), state.range(1), CV_8UC1);
    cv::Mat g(state.range(0), state.range(1), CV_8UC1);
    cv::Mat r(state.range(0), state.range(1), CV_8UC1);
    std::vector<cv::Mat> chs = {b, g, r};
    for (auto _ : state) {
        cv::Mat dst;
        cv::merge(chs, dst);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_merge_channels)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_merge_channels(benchmark::State& state) {
    Image<Gray> b(cv::Mat(state.range(0), state.range(1), CV_8UC1));
    Image<Gray> g(cv::Mat(state.range(0), state.range(1), CV_8UC1));
    Image<Gray> r(cv::Mat(state.range(0), state.range(1), CV_8UC1));
    for (auto _ : state)
        benchmark::DoNotOptimize(MergeChannels{}(b, g, r));
}
BENCHMARK(BM_improc_merge_channels)->Args({480, 640})->Args({1080, 1920});

// ── IntegralImage ─────────────────────────────────────────────────────────────

static void BM_raw_integral_image(benchmark::State& state) {
    cv::Mat src(480, 640, CV_8UC1);
    for (auto _ : state) {
        cv::Mat sum;
        cv::integral(src, sum);
        benchmark::DoNotOptimize(sum);
    }
}
BENCHMARK(BM_raw_integral_image);

static void BM_improc_integral_image(benchmark::State& state) {
    Image<Gray> img(cv::Mat(480, 640, CV_8UC1));
    IntegralImage op;
    for (auto _ : state)
        benchmark::DoNotOptimize(op(img));
}
BENCHMARK(BM_improc_integral_image);

// ── MinMaxLoc ─────────────────────────────────────────────────────────────────

static void BM_raw_minmax_loc(benchmark::State& state) {
    cv::Mat src(480, 640, CV_8UC1);
    double mn, mx;
    cv::Point mn_loc, mx_loc;
    for (auto _ : state) {
        cv::minMaxLoc(src, &mn, &mx, &mn_loc, &mx_loc);
        benchmark::DoNotOptimize(mx);
    }
}
BENCHMARK(BM_raw_minmax_loc);

static void BM_improc_minmax_loc(benchmark::State& state) {
    Image<Gray> img(cv::Mat(480, 640, CV_8UC1));
    MinMaxLoc op;
    for (auto _ : state)
        benchmark::DoNotOptimize(op(img));
}
BENCHMARK(BM_improc_minmax_loc);

// ── MeanStdDev ───────────────────────────────────────────────────────────────

static void BM_raw_mean_stddev(benchmark::State& state) {
    cv::Mat src(480, 640, CV_8UC1);
    cv::Scalar mean, stddev;
    for (auto _ : state) {
        cv::meanStdDev(src, mean, stddev);
        benchmark::DoNotOptimize(mean);
    }
}
BENCHMARK(BM_raw_mean_stddev);

static void BM_improc_mean_stddev(benchmark::State& state) {
    Image<Gray> img(cv::Mat(480, 640, CV_8UC1));
    MeanStdDev op;
    for (auto _ : state)
        benchmark::DoNotOptimize(op(img));
}
BENCHMARK(BM_improc_mean_stddev);

// ── CountNonZero ─────────────────────────────────────────────────────────────

static void BM_raw_count_nonzero(benchmark::State& state) {
    cv::Mat src(480, 640, CV_8UC1);
    for (auto _ : state)
        benchmark::DoNotOptimize(cv::countNonZero(src));
}
BENCHMARK(BM_raw_count_nonzero);

static void BM_improc_count_nonzero(benchmark::State& state) {
    Image<Gray> img(cv::Mat(480, 640, CV_8UC1));
    CountNonZero op;
    for (auto _ : state)
        benchmark::DoNotOptimize(op(img));
}
BENCHMARK(BM_improc_count_nonzero);

// ── Reduce (Sum, dim=0) ───────────────────────────────────────────────────────

static void BM_raw_reduce(benchmark::State& state) {
    cv::Mat src(480, 640, CV_8UC1);
    for (auto _ : state) {
        cv::Mat dst;
        cv::reduce(src, dst, 0, cv::REDUCE_SUM, CV_32SC1);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_reduce);

static void BM_improc_reduce(benchmark::State& state) {
    Image<Gray> img(cv::Mat(480, 640, CV_8UC1));
    Reduce op;
    for (auto _ : state)
        benchmark::DoNotOptimize(op(img));
}
BENCHMARK(BM_improc_reduce);
