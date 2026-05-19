// benchmarks/io/bench_video_file_capture.cpp
//
// Throughput: VideoFileCapture vs raw cv::VideoCapture frame-read loop.
// Creates a short synthetic .avi in /tmp on first use (skips if codec unavailable).
// Parametrized by frame count; measures total frames / elapsed time.
//
// Run: ./build/improc_benchmarks --benchmark_filter="video_file_capture"

#include <benchmark/benchmark.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include "improc/io/video_file_capture.hpp"

using namespace improc::io;

namespace {
// Creates a synthetic MJPG .avi with `frames` 640×480 BGR frames.
// Returns path on success, empty string if codec unavailable.
std::string make_temp_video(int frames = 60) {
    const std::string path = "/tmp/improc_bench_video.avi";
    cv::VideoWriter w(path, cv::VideoWriter::fourcc('M','J','P','G'),
                      30.0, cv::Size(640, 480));
    if (!w.isOpened()) return {};
    cv::Mat frame(480, 640, CV_8UC3);
    for (int i = 0; i < frames; ++i) {
        frame.setTo(cv::Scalar(i * 2, 100, 200));
        w.write(frame);
    }
    return path;
}

const std::string g_video_path = make_temp_video();
} // namespace

// ── Raw cv::VideoCapture ──────────────────────────────────────────────────────

static void BM_raw_video_file_capture(benchmark::State& state) {
    if (g_video_path.empty()) { state.SkipWithError("video file unavailable"); return; }
    for (auto _ : state) {
        cv::VideoCapture cap(g_video_path);
        cv::Mat frame;
        int count = 0;
        while (cap.read(frame)) { benchmark::DoNotOptimize(frame); ++count; }
        state.SetItemsProcessed(count);
    }
}
BENCHMARK(BM_raw_video_file_capture);

// ── improc::io::VideoFileCapture ─────────────────────────────────────────────

static void BM_improc_video_file_capture(benchmark::State& state) {
    if (g_video_path.empty()) { state.SkipWithError("video file unavailable"); return; }
    for (auto _ : state) {
        VideoFileCapture cap(g_video_path);
        cap.start();
        int count = 0;
        while (true) {
            auto f = cap.getFrame();
            if (!f) break;
            benchmark::DoNotOptimize(f);
            ++count;
        }
        cap.stop();
        state.SetItemsProcessed(count);
    }
}
BENCHMARK(BM_improc_video_file_capture);
