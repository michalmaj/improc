// benchmarks/core/bench_motion.cpp
//
// Motion ops — overhead (raw vs improc++ at 480×640) + throughput (parametrized).
// Overhead criterion: wrapper cost <= 5 ns per op.
//
// Build: cmake -DIMPROC_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
//              -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="conan_provider.cmake" -B build .
//        cmake --build build --target improc_benchmarks
// Run:   ./build/improc_benchmarks \
//          --benchmark_filter="farneback|dense_dis|sparse_lk|phase_correlate|camshift|meanshift"

#include <benchmark/benchmark.h>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

namespace {

Image<Gray> make_gray(int h, int w) {
    cv::Mat m(h, w, CV_8UC1);
    cv::randu(m, 0, 255);
    cv::GaussianBlur(m, m, {5, 5}, 1.5);
    return Image<Gray>(m);
}

Image<Gray> make_shifted(const Image<Gray>& src, int dx) {
    cv::Mat t = (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, 0);
    cv::Mat dst;
    cv::warpAffine(src.mat(), dst, t, src.mat().size());
    return Image<Gray>(dst);
}

Image<Float32> make_float(int h, int w) {
    cv::Mat m(h, w, CV_32FC1);
    cv::randu(m, 0.f, 1.f);
    return Image<Float32>(m);
}

std::vector<cv::Point2f> make_grid_pts(int h, int w) {
    std::vector<cv::Point2f> pts;
    for (int r = h / 10; r < h - h / 10; r += h / 10)
        for (int c = w / 10; c < w - w / 10; c += w / 10)
            pts.push_back({static_cast<float>(c), static_cast<float>(r)});
    return pts;
}

cv::Mat make_prob_map(int h, int w) {
    cv::Mat m(h, w, CV_8UC1, cv::Scalar(0));
    int cx = w / 2, cy = h / 2, r = std::min(h, w) / 4;
    m(cv::Rect(cx - r, cy - r, 2 * r, 2 * r)).setTo(cv::Scalar(255));
    return m;
}

} // namespace

// ── DenseFarnebackFlow — overhead ─────────────────────────────────────────────

static void BM_raw_dense_farneback(benchmark::State& state) {
    auto prev = make_gray(480, 640);
    auto next = make_shifted(prev, 5);
    for (auto _ : state) {
        cv::Mat flow;
        cv::calcOpticalFlowFarneback(prev.mat(), next.mat(), flow,
                                     0.5, 3, 15, 3, 5, 1.2, 0);
        benchmark::DoNotOptimize(flow);
    }
}
BENCHMARK(BM_raw_dense_farneback);

static void BM_improc_dense_farneback(benchmark::State& state) {
    auto prev = make_gray(480, 640);
    auto next = make_shifted(prev, 5);
    for (auto _ : state)
        benchmark::DoNotOptimize(DenseFarnebackFlow{}(prev, next));
}
BENCHMARK(BM_improc_dense_farneback);

// ── DenseFarnebackFlow — throughput ──────────────────────────────────────────

static void BM_dense_farneback(benchmark::State& state) {
    auto prev = make_gray(state.range(0), state.range(1));
    auto next = make_shifted(prev, 5);
    for (auto _ : state)
        benchmark::DoNotOptimize(DenseFarnebackFlow{}(prev, next));
}
BENCHMARK(BM_dense_farneback)->Args({480, 640})->Args({1080, 1920})->Iterations(5);

// ── DenseDISFlow — overhead ───────────────────────────────────────────────────

static void BM_raw_dense_dis_ultrafast(benchmark::State& state) {
    auto prev = make_gray(480, 640);
    auto next = make_shifted(prev, 5);
    auto dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_ULTRAFAST);
    for (auto _ : state) {
        cv::Mat flow;
        dis->calc(prev.mat(), next.mat(), flow);
        benchmark::DoNotOptimize(flow);
    }
}
BENCHMARK(BM_raw_dense_dis_ultrafast);

static void BM_improc_dense_dis_ultrafast(benchmark::State& state) {
    auto prev = make_gray(480, 640);
    auto next = make_shifted(prev, 5);
    DenseDISFlow op;
    op.preset(DenseDISFlow::Preset::UltraFast);
    for (auto _ : state)
        benchmark::DoNotOptimize(op(prev, next));
}
BENCHMARK(BM_improc_dense_dis_ultrafast);

// ── DenseDISFlow — throughput (3 presets) ────────────────────────────────────

static void BM_dense_dis_ultrafast(benchmark::State& state) {
    auto prev = make_gray(state.range(0), state.range(1));
    auto next = make_shifted(prev, 5);
    DenseDISFlow op;
    op.preset(DenseDISFlow::Preset::UltraFast);
    for (auto _ : state)
        benchmark::DoNotOptimize(op(prev, next));
}
BENCHMARK(BM_dense_dis_ultrafast)->Args({480, 640})->Args({1080, 1920})->Iterations(5);

static void BM_dense_dis_fast(benchmark::State& state) {
    auto prev = make_gray(state.range(0), state.range(1));
    auto next = make_shifted(prev, 5);
    DenseDISFlow op;
    op.preset(DenseDISFlow::Preset::Fast);
    for (auto _ : state)
        benchmark::DoNotOptimize(op(prev, next));
}
BENCHMARK(BM_dense_dis_fast)->Args({480, 640})->Args({1080, 1920})->Iterations(5);

static void BM_dense_dis_medium(benchmark::State& state) {
    auto prev = make_gray(state.range(0), state.range(1));
    auto next = make_shifted(prev, 5);
    DenseDISFlow op;
    op.preset(DenseDISFlow::Preset::Medium);
    for (auto _ : state)
        benchmark::DoNotOptimize(op(prev, next));
}
BENCHMARK(BM_dense_dis_medium)->Args({480, 640})->Args({1080, 1920})->Iterations(5);

// ── SparseLKFlow — overhead ───────────────────────────────────────────────────

static void BM_raw_sparse_lk(benchmark::State& state) {
    auto prev = make_gray(480, 640);
    auto next = make_shifted(prev, 5);
    auto pts  = make_grid_pts(480, 640);
    for (auto _ : state) {
        std::vector<cv::Point2f> next_pts;
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prev.mat(), next.mat(), pts,
                                  next_pts, status, err);
        benchmark::DoNotOptimize(next_pts);
    }
}
BENCHMARK(BM_raw_sparse_lk);

static void BM_improc_sparse_lk(benchmark::State& state) {
    auto prev = make_gray(480, 640);
    auto next = make_shifted(prev, 5);
    auto pts  = make_grid_pts(480, 640);
    SparseLKFlow op;
    for (auto _ : state)
        benchmark::DoNotOptimize(op(prev, next, pts));
}
BENCHMARK(BM_improc_sparse_lk);

// ── SparseLKFlow — throughput ─────────────────────────────────────────────────

static void BM_sparse_lk(benchmark::State& state) {
    auto prev = make_gray(state.range(0), state.range(1));
    auto next = make_shifted(prev, 5);
    auto pts  = make_grid_pts(state.range(0), state.range(1));
    SparseLKFlow op;
    for (auto _ : state)
        benchmark::DoNotOptimize(op(prev, next, pts));
}
BENCHMARK(BM_sparse_lk)->Args({480, 640})->Args({1080, 1920})->Iterations(5);

// ── PhaseCorrelate — overhead ─────────────────────────────────────────────────

static void BM_raw_phase_correlate(benchmark::State& state) {
    auto prev_f = make_float(480, 640);
    auto next_f = make_float(480, 640);
    cv::Mat hann;
    cv::createHanningWindow(hann, prev_f.mat().size(), CV_32F);
    double response;
    for (auto _ : state) {
        auto shift = cv::phaseCorrelate(prev_f.mat(), next_f.mat(), hann, &response);
        benchmark::DoNotOptimize(shift);
        benchmark::DoNotOptimize(response);
    }
}
BENCHMARK(BM_raw_phase_correlate);

static void BM_improc_phase_correlate(benchmark::State& state) {
    auto prev_f = make_float(480, 640);
    auto next_f = make_float(480, 640);
    PhaseCorrelate op;
    for (auto _ : state)
        benchmark::DoNotOptimize(op(prev_f, next_f));
}
BENCHMARK(BM_improc_phase_correlate);

// ── PhaseCorrelate — throughput ───────────────────────────────────────────────

static void BM_phase_correlate(benchmark::State& state) {
    auto prev_f = make_float(state.range(0), state.range(1));
    auto next_f = make_float(state.range(0), state.range(1));
    PhaseCorrelate op;
    for (auto _ : state)
        benchmark::DoNotOptimize(op(prev_f, next_f));
}
BENCHMARK(BM_phase_correlate)->Args({480, 640})->Args({1080, 1920})->Iterations(5);

// ── CamShift — overhead ───────────────────────────────────────────────────────

static void BM_raw_camshift(benchmark::State& state) {
    cv::Mat back_proj = make_prob_map(480, 640);
    const cv::Rect init_window(240, 160, 160, 160);
    auto criteria = cv::TermCriteria(
        cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 1.0);
    for (auto _ : state) {
        state.PauseTiming();
        cv::Rect window = init_window;
        state.ResumeTiming();
        cv::CamShift(back_proj, window, criteria);
        benchmark::DoNotOptimize(window);
    }
}
BENCHMARK(BM_raw_camshift);

static void BM_improc_camshift(benchmark::State& state) {
    Image<Gray> back_proj(make_prob_map(480, 640));
    const cv::Rect init_window(240, 160, 160, 160);
    CamShift op;
    for (auto _ : state) {
        state.PauseTiming();
        cv::Rect window = init_window;
        state.ResumeTiming();
        benchmark::DoNotOptimize(op(back_proj, window));
    }
}
BENCHMARK(BM_improc_camshift);

// ── CamShift — throughput ─────────────────────────────────────────────────────

static void BM_camshift(benchmark::State& state) {
    Image<Gray> back_proj(make_prob_map(state.range(0), state.range(1)));
    const cv::Rect init_window(state.range(1) / 4, state.range(0) / 4,
                                state.range(1) / 4, state.range(0) / 4);
    CamShift op;
    for (auto _ : state) {
        state.PauseTiming();
        cv::Rect window = init_window;
        state.ResumeTiming();
        benchmark::DoNotOptimize(op(back_proj, window));
    }
}
BENCHMARK(BM_camshift)->Args({480, 640})->Args({1080, 1920})->Iterations(5);

// ── MeanShift — overhead ──────────────────────────────────────────────────────

static void BM_raw_meanshift(benchmark::State& state) {
    cv::Mat back_proj = make_prob_map(480, 640);
    const cv::Rect init_window(240, 160, 160, 160);
    auto criteria = cv::TermCriteria(
        cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 1.0);
    for (auto _ : state) {
        state.PauseTiming();
        cv::Rect window = init_window;
        state.ResumeTiming();
        cv::meanShift(back_proj, window, criteria);
        benchmark::DoNotOptimize(window);
    }
}
BENCHMARK(BM_raw_meanshift);

static void BM_improc_meanshift(benchmark::State& state) {
    Image<Gray> back_proj(make_prob_map(480, 640));
    const cv::Rect init_window(240, 160, 160, 160);
    MeanShift op;
    for (auto _ : state) {
        state.PauseTiming();
        cv::Rect window = init_window;
        state.ResumeTiming();
        benchmark::DoNotOptimize(op(back_proj, window));
    }
}
BENCHMARK(BM_improc_meanshift);

// ── MeanShift — throughput ────────────────────────────────────────────────────

static void BM_meanshift(benchmark::State& state) {
    Image<Gray> back_proj(make_prob_map(state.range(0), state.range(1)));
    const cv::Rect init_window(state.range(1) / 4, state.range(0) / 4,
                                state.range(1) / 4, state.range(0) / 4);
    MeanShift op;
    for (auto _ : state) {
        state.PauseTiming();
        cv::Rect window = init_window;
        state.ResumeTiming();
        benchmark::DoNotOptimize(op(back_proj, window));
    }
}
BENCHMARK(BM_meanshift)->Args({480, 640})->Args({1080, 1920})->Iterations(5);
