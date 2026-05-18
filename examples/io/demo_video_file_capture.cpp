// examples/io/demo_video_file_capture.cpp
//
// Usage: ./demo_video_file_capture <video_file>
// Reads the video file through FramePipeline, resizes each frame to 320x240,
// and reports fps at the end. No display window — pure processing demo.
//
#include "improc/io/video_file_capture.hpp"
#include "improc/threading/frame_pipeline.hpp"
#include "improc/threading/thread_pool.hpp"
#include "improc/core/pipeline.hpp"
#include <chrono>
#include <iostream>

using improc::io::VideoFileCapture;
using improc::io::CameraFrame;
using improc::threading::FramePipeline;
using improc::threading::ThreadPool;
using namespace improc::core;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_file>\n";
        return 1;
    }

    VideoFileCapture cap(argv[1]);
    cap.start();

    ThreadPool pool(4);
    FramePipeline<cv::Mat> pipeline(cap, pool);
    pipeline.start([](CameraFrame frame) -> cv::Mat {
        if (!frame.rgb) return {};
        return (*frame.rgb | Resize{}.width(320).height(240)).mat().clone();
    });

    int frames = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (true) {
        if (auto result = pipeline.tryPop()) {
            if (!result->empty()) ++frames;
        } else {
            auto probe = cap.getFrame();
            if (!probe && probe.error().code == improc::Error::Code::EndOfFile) break;
        }
    }
    pipeline.stop();

    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::cout << "Processed " << frames << " frames in " << secs << "s"
              << "  (" << (secs > 0 ? frames / secs : 0.0) << " fps)\n";
    return 0;
}
