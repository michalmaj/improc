// examples/io/demo_unified_camera.cpp
#include "improc/io/io.hpp"
#include "improc/threading/frame_pipeline.hpp"
#include "improc/threading/thread_pool.hpp"
#include "improc/core/pipeline.hpp"
#include <iostream>
#include <chrono>
#include <opencv2/highgui.hpp>

using namespace improc::io;
using namespace improc::core;
using improc::threading::FramePipeline;
using improc::threading::ThreadPool;

int main(int argc, char* argv[]) {
    AnyCameraSource source;
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg.starts_with("rtsp://") || arg.starts_with("http://") || arg.starts_with("https://")) {
            std::cout << "Using IPCameraCapture: " << arg << "\n";
            source = AnyCameraSource::make<IPCameraCapture>(arg);
        } else {
            std::cout << "Argument not a URL; using WebcamCapture(0)\n";
            source = AnyCameraSource::make<WebcamCapture>(0);
        }
    } else {
        std::cout << "No argument; using WebcamCapture(0)\n";
        source = AnyCameraSource::make<WebcamCapture>(0);
    }

    ThreadPool pool(4);
    FramePipeline<cv::Mat> pipeline(source, pool);

    pipeline.start([](CameraFrame frame) -> cv::Mat {
        if (!frame.rgb) return {};
        Image<BGR> resized = *frame.rgb | Resize{}.width(320).height(240);
        return resized.mat().clone();
    });

    std::cout << "Press any key to stop...\n";
    int frames = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (true) {
        if (auto result = pipeline.tryPop()) {
            if (!result->empty()) {
                ++frames;
                cv::imshow("Unified Camera Demo", *result);
            }
        }
        if (cv::waitKey(1) >= 0) break;
    }
    pipeline.stop();

    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::cout << "Processed " << frames << " frames in " << secs << "s"
              << "  (" << (secs > 0 ? frames / secs : 0.0) << " fps)\n";
    cv::destroyAllWindows();
    return 0;
}
