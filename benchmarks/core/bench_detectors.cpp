// benchmarks/core/bench_detectors.cpp
//
// Detector ops — overhead (raw vs improc++ at 480×640) + throughput (parametrized).
// Overhead criterion: wrapper cost <= 5 ns per op.
// Face benchmarks are model-gated; skipped gracefully when .onnx files are absent.
//
// Build: cmake -DIMPROC_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
//              -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="conan_provider.cmake" \
//              -DCONAN_COMMAND=/opt/anaconda3/envs/conan_cv/bin/conan -B build .
//        cmake --build build --target improc_benchmarks --parallel 2
// Run:   ./build/improc_benchmarks \
//          --benchmark_filter="detect_fast|detect_blob|detect_mser|detect_lines|detect_qr|detect_barcode|detect_face|recognize_face"

#include <benchmark/benchmark.h>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

namespace {

static constexpr const char* kFaceModel  = "tests/core/testdata/face_detection_yunet.onnx";
static constexpr const char* kRecogModel = "tests/core/testdata/face_recognition_sface.onnx";

Image<Gray> make_textured_gray(int h, int w) {
    cv::Mat m(h, w, CV_8UC1);
    cv::randu(m, 0, 255);
    cv::GaussianBlur(m, m, {5, 5}, 1.5);
    return Image<Gray>(m);
}

Image<Gray> make_circles_gray(int h, int w) {
    cv::Mat m(h, w, CV_8UC1, cv::Scalar(0));
    cv::circle(m, {w / 4,     h / 2}, 20, cv::Scalar(255), -1);
    cv::circle(m, {w / 2,     h / 2}, 20, cv::Scalar(255), -1);
    cv::circle(m, {3 * w / 4, h / 2}, 20, cv::Scalar(255), -1);
    return Image<Gray>(m);
}

Image<Gray> make_nested_rects_gray(int h, int w) {
    cv::Mat m(h, w, CV_8UC1, cv::Scalar(128));
    cv::rectangle(m, cv::Rect(w/8, h/8, 3*w/4, 3*h/4), cv::Scalar(30),  4);
    cv::rectangle(m, cv::Rect(w/4, h/4,   w/2,   h/2), cv::Scalar(220), 4);
    return Image<Gray>(m);
}

Image<Gray> make_lines_gray(int h, int w) {
    cv::Mat m(h, w, CV_8UC1, cv::Scalar(255));
    cv::line(m, {0,     0    }, {w - 1, h - 1}, cv::Scalar(0), 3);
    cv::line(m, {0,     h / 2}, {w - 1, h / 2}, cv::Scalar(0), 3);
    cv::line(m, {w / 2, 0    }, {w / 2, h - 1}, cv::Scalar(0), 3);
    cv::line(m, {w - 1, 0    }, {0,     h - 1}, cv::Scalar(0), 3);
    return Image<Gray>(m);
}

Image<BGR> make_qr_bgr(int h, int w) {
    cv::Mat qr_gray;
    cv::QRCodeEncoder::create()->encode("BENCH", qr_gray);
    cv::Mat resized, bgr;
    cv::resize(qr_gray, resized, {w, h}, 0, 0, cv::INTER_NEAREST);
    cv::cvtColor(resized, bgr, cv::COLOR_GRAY2BGR);
    return Image<BGR>(bgr);
}

Image<BGR> make_blank_bgr(int h, int w) {
    return Image<BGR>(cv::Mat(h, w, CV_8UC3, cv::Scalar(255, 255, 255)));
}

} // namespace

// ── DetectFAST — overhead ─────────────────────────────────────────────────────

static void BM_raw_detect_fast(benchmark::State& state) {
    auto img = make_textured_gray(480, 640);
    for (auto _ : state) {
        std::vector<cv::KeyPoint> kps;
        cv::FAST(img.mat(), kps, 10, true);
        benchmark::DoNotOptimize(kps);
    }
}
BENCHMARK(BM_raw_detect_fast);

static void BM_improc_detect_fast(benchmark::State& state) {
    auto img = make_textured_gray(480, 640);
    // DetectFAST defaults: threshold=10, non_max_suppression=true — matches raw above
    for (auto _ : state)
        benchmark::DoNotOptimize(DetectFAST{}(img));
}
BENCHMARK(BM_improc_detect_fast);

// ── DetectFAST — throughput ───────────────────────────────────────────────────

static void BM_detect_fast(benchmark::State& state) {
    auto img = make_textured_gray(state.range(0), state.range(1));
    for (auto _ : state)
        benchmark::DoNotOptimize(DetectFAST{}(img));
}
BENCHMARK(BM_detect_fast)->Args({480, 640})->Args({720, 1280})->Iterations(5);

// ── DetectBlob — overhead ─────────────────────────────────────────────────────

static void BM_raw_detect_blob(benchmark::State& state) {
    auto img      = make_circles_gray(480, 640);
    auto detector = cv::SimpleBlobDetector::create();
    for (auto _ : state) {
        std::vector<cv::KeyPoint> kps;
        detector->detect(img.mat(), kps);
        benchmark::DoNotOptimize(kps);
    }
}
BENCHMARK(BM_raw_detect_blob);

static void BM_improc_detect_blob(benchmark::State& state) {
    auto img = make_circles_gray(480, 640);
    // DetectBlob default-constructs params — same defaults as SimpleBlobDetector::create()
    for (auto _ : state)
        benchmark::DoNotOptimize(DetectBlob{}(img));
}
BENCHMARK(BM_improc_detect_blob);

// ── DetectBlob — throughput ───────────────────────────────────────────────────

static void BM_detect_blob(benchmark::State& state) {
    auto img = make_circles_gray(state.range(0), state.range(1));
    for (auto _ : state)
        benchmark::DoNotOptimize(DetectBlob{}(img));
}
BENCHMARK(BM_detect_blob)->Args({480, 640})->Args({720, 1280})->Iterations(5);

// ── DetectMSER — overhead ─────────────────────────────────────────────────────

static void BM_raw_detect_mser(benchmark::State& state) {
    auto img = make_nested_rects_gray(480, 640);
    for (auto _ : state) {
        std::vector<std::vector<cv::Point>> regions;
        std::vector<cv::Rect> bboxes;
        cv::MSER::create()->detectRegions(img.mat(), regions, bboxes);
        benchmark::DoNotOptimize(regions);
    }
}
BENCHMARK(BM_raw_detect_mser);

static void BM_improc_detect_mser(benchmark::State& state) {
    auto img = make_nested_rects_gray(480, 640);
    // DetectMSER defaults: delta=5, min_area=60, max_area=14400 — matches MSER::create() defaults
    for (auto _ : state)
        benchmark::DoNotOptimize(DetectMSER{}(img));
}
BENCHMARK(BM_improc_detect_mser);

// ── DetectMSER — throughput ───────────────────────────────────────────────────

static void BM_detect_mser(benchmark::State& state) {
    auto img = make_nested_rects_gray(state.range(0), state.range(1));
    for (auto _ : state)
        benchmark::DoNotOptimize(DetectMSER{}(img));
}
BENCHMARK(BM_detect_mser)->Args({480, 640})->Args({720, 1280})->Iterations(5);

// ── DetectLines — overhead ────────────────────────────────────────────────────

static void BM_raw_detect_lines(benchmark::State& state) {
    auto img = make_lines_gray(480, 640);
    auto lsd  = cv::createLineSegmentDetector(cv::LSD_REFINE_STD, 0.8, 0.6);
    for (auto _ : state) {
        cv::Mat lines_mat;
        lsd->detect(img.mat(), lines_mat);
        benchmark::DoNotOptimize(lines_mat);
    }
}
BENCHMARK(BM_raw_detect_lines);

static void BM_improc_detect_lines(benchmark::State& state) {
    auto img = make_lines_gray(480, 640);
    // DetectLines defaults: scale=0.8, sigma_scale=0.6 — matches LSD_REFINE_STD above
    for (auto _ : state)
        benchmark::DoNotOptimize(DetectLines{}(img));
}
BENCHMARK(BM_improc_detect_lines);

// ── DetectLines — throughput ──────────────────────────────────────────────────

static void BM_detect_lines(benchmark::State& state) {
    auto img = make_lines_gray(state.range(0), state.range(1));
    for (auto _ : state)
        benchmark::DoNotOptimize(DetectLines{}(img));
}
BENCHMARK(BM_detect_lines)->Args({480, 640})->Args({720, 1280})->Iterations(5);
