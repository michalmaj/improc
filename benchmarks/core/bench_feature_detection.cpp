// benchmarks/core/bench_feature_detection.cpp
//
// Feature detection throughput: raw OpenCV vs improc++ for ORB, SIFT, AKAZE.
// Parametrized: ->Args({H, W}) with {480,640} and {1080,1920}.
//
// Run: ./build/improc_benchmarks --benchmark_filter="orb|sift|akaze|match_bf|match_flann"

#include <benchmark/benchmark.h>
#include <opencv2/features2d.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

// ── ORB detect ───────────────────────────────────────────────────────────────

static void BM_raw_orb_detect(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC1);
    cv::randu(src, cv::Scalar(0), cv::Scalar(255));
    auto orb = cv::ORB::create(500);
    std::vector<cv::KeyPoint> kps;
    for (auto _ : state) {
        kps.clear();
        orb->detect(src, kps);
        benchmark::DoNotOptimize(kps);
    }
}
BENCHMARK(BM_raw_orb_detect)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_orb_detect(benchmark::State& state) {
    cv::Mat raw(state.range(0), state.range(1), CV_8UC1);
    cv::randu(raw, cv::Scalar(0), cv::Scalar(255));
    Image<Gray> img(raw);
    for (auto _ : state)
        benchmark::DoNotOptimize(img | DetectORB{}.max_features(500));
}
BENCHMARK(BM_improc_orb_detect)->Args({480, 640})->Args({1080, 1920});

// ── SIFT detect ──────────────────────────────────────────────────────────────

static void BM_raw_sift_detect(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC1);
    cv::randu(src, cv::Scalar(0), cv::Scalar(255));
    auto sift = cv::SIFT::create(500);
    std::vector<cv::KeyPoint> kps;
    for (auto _ : state) {
        kps.clear();
        sift->detect(src, kps);
        benchmark::DoNotOptimize(kps);
    }
}
BENCHMARK(BM_raw_sift_detect)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_sift_detect(benchmark::State& state) {
    cv::Mat raw(state.range(0), state.range(1), CV_8UC1);
    cv::randu(raw, cv::Scalar(0), cv::Scalar(255));
    Image<Gray> img(raw);
    for (auto _ : state)
        benchmark::DoNotOptimize(img | DetectSIFT{}.max_features(500));
}
BENCHMARK(BM_improc_sift_detect)->Args({480, 640})->Args({1080, 1920});

// ── AKAZE detect ─────────────────────────────────────────────────────────────

static void BM_raw_akaze_detect(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC1);
    cv::randu(src, cv::Scalar(0), cv::Scalar(255));
    auto akaze = cv::AKAZE::create();
    std::vector<cv::KeyPoint> kps;
    for (auto _ : state) {
        kps.clear();
        akaze->detect(src, kps);
        benchmark::DoNotOptimize(kps);
    }
}
BENCHMARK(BM_raw_akaze_detect)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_akaze_detect(benchmark::State& state) {
    cv::Mat raw(state.range(0), state.range(1), CV_8UC1);
    cv::randu(raw, cv::Scalar(0), cv::Scalar(255));
    Image<Gray> img(raw);
    for (auto _ : state)
        benchmark::DoNotOptimize(img | DetectAKAZE{});
}
BENCHMARK(BM_improc_akaze_detect)->Args({480, 640})->Args({1080, 1920});

// ── ORB describe ─────────────────────────────────────────────────────────────

static void BM_raw_orb_describe(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC1);
    cv::randu(src, cv::Scalar(0), cv::Scalar(255));
    auto orb = cv::ORB::create(500);
    std::vector<cv::KeyPoint> kps;
    orb->detect(src, kps);
    cv::Mat desc;
    for (auto _ : state) {
        orb->compute(src, kps, desc);
        benchmark::DoNotOptimize(desc);
    }
}
BENCHMARK(BM_raw_orb_describe)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_orb_describe(benchmark::State& state) {
    cv::Mat raw(state.range(0), state.range(1), CV_8UC1);
    cv::randu(raw, cv::Scalar(0), cv::Scalar(255));
    Image<Gray> img(raw);
    auto kps = img | DetectORB{}.max_features(500);
    for (auto _ : state)
        benchmark::DoNotOptimize(img | DescribeORB{kps});
}
BENCHMARK(BM_improc_orb_describe)->Args({480, 640})->Args({1080, 1920});

// ── SIFT describe ────────────────────────────────────────────────────────────

static void BM_raw_sift_describe(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC1);
    cv::randu(src, cv::Scalar(0), cv::Scalar(255));
    auto sift = cv::SIFT::create(500);
    std::vector<cv::KeyPoint> kps;
    sift->detect(src, kps);
    cv::Mat desc;
    for (auto _ : state) {
        sift->compute(src, kps, desc);
        benchmark::DoNotOptimize(desc);
    }
}
BENCHMARK(BM_raw_sift_describe)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_sift_describe(benchmark::State& state) {
    cv::Mat raw(state.range(0), state.range(1), CV_8UC1);
    cv::randu(raw, cv::Scalar(0), cv::Scalar(255));
    Image<Gray> img(raw);
    auto kps = img | DetectSIFT{}.max_features(500);
    for (auto _ : state)
        benchmark::DoNotOptimize(img | DescribeSIFT{kps});
}
BENCHMARK(BM_improc_sift_describe)->Args({480, 640})->Args({1080, 1920});

// ── MatchBF (ORB descriptors) ─────────────────────────────────────────────────

static void BM_raw_match_bf(benchmark::State& state) {
    cv::Mat src1(480, 640, CV_8UC1), src2(480, 640, CV_8UC1);
    cv::randu(src1, cv::Scalar(0), cv::Scalar(255));
    cv::randu(src2, cv::Scalar(0), cv::Scalar(255));
    auto orb = cv::ORB::create(500);
    std::vector<cv::KeyPoint> kps1, kps2;
    cv::Mat desc1, desc2;
    orb->detectAndCompute(src1, cv::noArray(), kps1, desc1);
    orb->detectAndCompute(src2, cv::noArray(), kps2, desc2);
    auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    for (auto _ : state) {
        matches.clear();
        matcher->match(desc1, desc2, matches);
        benchmark::DoNotOptimize(matches);
    }
}
BENCHMARK(BM_raw_match_bf);

static void BM_improc_match_bf(benchmark::State& state) {
    cv::Mat raw1(480, 640, CV_8UC1), raw2(480, 640, CV_8UC1);
    cv::randu(raw1, cv::Scalar(0), cv::Scalar(255));
    cv::randu(raw2, cv::Scalar(0), cv::Scalar(255));
    Image<Gray> img1(raw1);
    Image<Gray> img2(raw2);
    auto desc1 = img1 | DescribeORB{ img1 | DetectORB{}.max_features(500) };
    auto desc2 = img2 | DescribeORB{ img2 | DetectORB{}.max_features(500) };
    for (auto _ : state)
        benchmark::DoNotOptimize(MatchBF{desc1, desc2}.cross_check(true)());
}
BENCHMARK(BM_improc_match_bf);

// ── MatchFlann (SIFT descriptors) ─────────────────────────────────────────────

static void BM_raw_match_flann(benchmark::State& state) {
    cv::Mat src1(480, 640, CV_8UC1), src2(480, 640, CV_8UC1);
    cv::randu(src1, cv::Scalar(0), cv::Scalar(255));
    cv::randu(src2, cv::Scalar(0), cv::Scalar(255));
    auto sift = cv::SIFT::create(500);
    std::vector<cv::KeyPoint> kps1, kps2;
    cv::Mat desc1, desc2;
    sift->detectAndCompute(src1, cv::noArray(), kps1, desc1);
    sift->detectAndCompute(src2, cv::noArray(), kps2, desc2);
    cv::FlannBasedMatcher flann;
    std::vector<std::vector<cv::DMatch>> knn;
    for (auto _ : state) {
        knn.clear();
        flann.knnMatch(desc1, desc2, knn, 2);
        benchmark::DoNotOptimize(knn);
    }
}
BENCHMARK(BM_raw_match_flann);

static void BM_improc_match_flann(benchmark::State& state) {
    cv::Mat raw1(480, 640, CV_8UC1), raw2(480, 640, CV_8UC1);
    cv::randu(raw1, cv::Scalar(0), cv::Scalar(255));
    cv::randu(raw2, cv::Scalar(0), cv::Scalar(255));
    Image<Gray> img1(raw1);
    Image<Gray> img2(raw2);
    auto desc1 = img1 | DescribeSIFT{ img1 | DetectSIFT{}.max_features(500) };
    auto desc2 = img2 | DescribeSIFT{ img2 | DetectSIFT{}.max_features(500) };
    for (auto _ : state)
        benchmark::DoNotOptimize(MatchFlann{desc1, desc2}.ratio_threshold(0.75f)());
}
BENCHMARK(BM_improc_match_flann);

// ── ORB end-to-end: detect → describe → match ────────────────────────────────

static void BM_improc_orb_e2e(benchmark::State& state) {
    cv::Mat raw1(state.range(0), state.range(1), CV_8UC1);
    cv::Mat raw2(state.range(0), state.range(1), CV_8UC1);
    cv::randu(raw1, cv::Scalar(0), cv::Scalar(255));
    cv::randu(raw2, cv::Scalar(0), cv::Scalar(255));
    Image<Gray> img1(raw1);
    Image<Gray> img2(raw2);
    for (auto _ : state) {
        auto kps1  = img1 | DetectORB{}.max_features(500);
        auto kps2  = img2 | DetectORB{}.max_features(500);
        auto desc1 = img1 | DescribeORB{kps1};
        auto desc2 = img2 | DescribeORB{kps2};
        benchmark::DoNotOptimize(MatchBF{desc1, desc2}.cross_check(true)());
    }
}
BENCHMARK(BM_improc_orb_e2e)->Args({480, 640})->Args({1080, 1920});

static void BM_improc_sift_e2e(benchmark::State& state) {
    cv::Mat raw1(state.range(0), state.range(1), CV_8UC1);
    cv::Mat raw2(state.range(0), state.range(1), CV_8UC1);
    cv::randu(raw1, cv::Scalar(0), cv::Scalar(255));
    cv::randu(raw2, cv::Scalar(0), cv::Scalar(255));
    Image<Gray> img1(raw1);
    Image<Gray> img2(raw2);
    for (auto _ : state) {
        auto kps1  = img1 | DetectSIFT{}.max_features(500);
        auto kps2  = img2 | DetectSIFT{}.max_features(500);
        auto desc1 = img1 | DescribeSIFT{kps1};
        auto desc2 = img2 | DescribeSIFT{kps2};
        benchmark::DoNotOptimize(MatchFlann{desc1, desc2}.ratio_threshold(0.75f)());
    }
}
BENCHMARK(BM_improc_sift_e2e)->Args({480, 640})->Args({1080, 1920});
