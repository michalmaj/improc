// benchmarks/core/bench_pipeline.cpp
//
// Throughput suite: reference ML preprocessing pipeline on 1920×1080 input.
// Acceptance criterion: >= 30 FPS single thread (<=33.3ms per frame).
//
// Build: cmake -DIMPROC_BENCHMARKS=ON ...
// Run:   ./build/improc_benchmarks --benchmark_filter="pipeline"

#include <benchmark/benchmark.h>
#include <opencv2/imgproc.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

static void BM_improc_ml_pipeline(benchmark::State& state) {
    cv::Mat raw(1080, 1920, CV_8UC3);
    cv::randu(raw, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    Image<BGR> src(raw);

    for (auto _ : state) {
        benchmark::DoNotOptimize(
            src
            | Resize{}.width(224).height(224)
            | CLAHE{}.clip_limit(2.0)
            | GaussianBlur{}.kernel_size(3)
            | ToFloat32C3{}
            | NormalizeTo{0.0f, 1.0f}
        );
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_improc_ml_pipeline)->Unit(benchmark::kMillisecond);

static void BM_raw_ml_pipeline(benchmark::State& state) {
    cv::Mat raw(1080, 1920, CV_8UC3);
    cv::randu(raw, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    cv::Mat resized, lab, blurred, float_img, normalized;
    std::vector<cv::Mat> lab_planes;
    for (auto _ : state) {

        cv::resize(raw, resized, cv::Size(224, 224));

        cv::cvtColor(resized, lab, cv::COLOR_BGR2Lab);
        cv::split(lab, lab_planes);
        auto clahe_obj = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe_obj->apply(lab_planes[0], lab_planes[0]);
        cv::merge(lab_planes, lab);
        cv::cvtColor(lab, resized, cv::COLOR_Lab2BGR);

        cv::GaussianBlur(resized, blurred, cv::Size(3, 3), 0);
        blurred.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
        double mn, mx;
        cv::minMaxLoc(float_img.reshape(1), &mn, &mx);
        float scale = (mx > mn) ? static_cast<float>(1.0 / (mx - mn)) : 1.0f;
        float shift = static_cast<float>(-mn * scale);
        float_img.convertTo(normalized, CV_32FC3, scale, shift);

        benchmark::DoNotOptimize(normalized);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_raw_ml_pipeline)->Unit(benchmark::kMillisecond);
