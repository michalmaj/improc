// examples/threading/demo_frame_pipeline.cpp
#include "improc/threading/frame_pipeline.hpp"
#include "improc/threading/thread_pool.hpp"
#include "improc/io/webcam_capture.hpp"
#include "improc/core/pipeline.hpp"
#include <chrono>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <thread>

using improc::io::WebcamCapture;
using improc::io::CameraFrame;
using improc::threading::FramePipeline;
using improc::threading::ThreadPool;
using namespace improc::core;

int main() {
    WebcamCapture camera(0);
    camera.start();
    std::cout << "Camera opened. Warming up...\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    ThreadPool pool(4);

    // Pipeline: capture → Resize(320x240) → display result
    std::cout << "Starting FramePipeline<cv::Mat>: Resize to 320x240\n";
    FramePipeline<cv::Mat> pipeline(camera, pool);
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
                cv::imshow("FramePipeline — Resize 320x240", *result);
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

    // Pipeline with typed result: extract frame height
    std::cout << "\nStarting FramePipeline<int>: extract frame height\n";
    FramePipeline<int> height_pipeline(camera, pool);
    height_pipeline.start([](CameraFrame frame) {
        return frame.rgb ? frame.rgb->mat().rows : 0;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    if (auto h = height_pipeline.tryPop())
        std::cout << "Frame height: " << *h << " px\n";
    height_pipeline.stop();

    std::cout << "\nDone.\n";
    return 0;
}
