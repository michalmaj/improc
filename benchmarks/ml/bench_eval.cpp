// benchmarks/ml/bench_eval.cpp
//
// Eval accumulator throughput: ClassEval, DetectionEval, SegEval.
// DetectionEval parametrized by number of detections per update() call.
// SegEval parametrized by mask size.
//
// Run: ./build/improc_benchmarks --benchmark_filter="eval"

#include <benchmark/benchmark.h>
#include "improc/ml/eval/classification.hpp"
#include "improc/ml/eval/eval.hpp"
#include "improc/ml/result_types.hpp"
#include "improc/ml/annotated.hpp"

using namespace improc::ml;
using namespace improc::core;

namespace {
std::vector<Detection> make_detections(int n) {
    std::vector<Detection> dets;
    dets.reserve(n);
    for (int i = 0; i < n; ++i) {
        Detection d;
        d.box        = cv::Rect2f(10.f * i, 10.f * i, 50.f, 50.f);
        d.class_id   = i % 3;
        d.confidence = 0.9f - 0.01f * i;
        d.label      = "cls" + std::to_string(i % 3);
        dets.push_back(d);
    }
    return dets;
}

std::vector<BBox> make_gt_boxes(int n) {
    std::vector<BBox> boxes;
    boxes.reserve(n);
    for (int i = 0; i < n; ++i) {
        BBox b;
        b.box      = cv::Rect2f(10.f * i, 10.f * i, 55.f, 55.f);
        b.class_id = i % 3;
        b.label    = "cls" + std::to_string(i % 3);
        boxes.push_back(b);
    }
    return boxes;
}
} // namespace

// ── ClassEval::update() ──────────────────────────────────────────────────────

static void BM_class_eval_update(benchmark::State& state) {
    ClassEval eval{};
    eval.class_names({"cat", "dog", "bird"});
    int i = 0;
    for (auto _ : state) {
        eval.update(i % 3, (i + 1) % 3);
        ++i;
        benchmark::DoNotOptimize(i);
    }
}
BENCHMARK(BM_class_eval_update);

// ── DetectionEval::update() — parametrized by detection count ────────────────

static void BM_detection_eval_update(benchmark::State& state) {
    int n = static_cast<int>(state.range(0));
    auto preds = make_detections(n);
    auto gts   = make_gt_boxes(n);
    for (auto _ : state) {
        DetectionEval eval{};
        eval.update(preds, gts);
        benchmark::DoNotOptimize(preds);
    }
}
BENCHMARK(BM_detection_eval_update)->Arg(5)->Arg(20)->Arg(50);

// ── SegEval::update() — parametrized by mask size ────────────────────────────

static void BM_seg_eval_update(benchmark::State& state) {
    int h = state.range(0), w = state.range(1);
    Image<Gray> pred(cv::Mat(h, w, CV_8UC1, cv::Scalar(1)));
    Image<Gray> gt(cv::Mat(h, w, CV_8UC1, cv::Scalar(1)));
    SegEval eval{};
    eval.num_classes(21);
    for (auto _ : state) {
        eval.update(pred, gt);
        benchmark::DoNotOptimize(pred);
    }
}
BENCHMARK(BM_seg_eval_update)->Args({224, 224})->Args({640, 640});
