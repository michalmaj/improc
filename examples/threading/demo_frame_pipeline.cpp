//
// Created by Michał Maj on 12/04/2026.
//

#include "improc/threading/frame_pipeline.hpp"
#include "improc/threading/thread_pool.hpp"
#include "improc/io/camera_capture.hpp"
#include "improc/core/pipeline.hpp"
#include <chrono>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <thread>

using improc::io::CameraCapture;
using improc::threading::FramePipeline;
using improc::threading::ThreadPool;
using namespace improc::core;

int main() {
    CameraCapture camera(0);
    std::cout << "Camera opened. Warming up...\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    ThreadPool pool(4);

    // --- 1. Pipeline: capture → Resize(320x240) → display result ---
    std::cout << "Starting FramePipeline<cv::Mat>: Resize to 320x240\n";
    FramePipeline<cv::Mat> pipeline(camera, pool);
    pipeline.start([](cv::Mat frame) -> cv::Mat {
        Image<BGR> img(frame);
        Image<BGR> resized = img | Resize{}.width(320).height(240);
        return resized.mat().clone();
    });

    std::cout << "Press any key to stop...\n";
    int frames = 0;
    const auto start = std::chrono::steady_clock::now();
    while (true) {
        if (auto result = pipeline.tryPop()) {
            ++frames;
            cv::imshow("FramePipeline — Resize 320x240", *result);
        }
        if (cv::waitKey(1) >= 0) break;
    }
    pipeline.stop();
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const double secs = std::chrono::duration<double>(elapsed).count();
    std::cout << "Processed " << frames << " frames in " << secs << "s"
              << "  (" << frames / secs << " fps)\n";

    cv::destroyAllWindows();

    // --- 2. Pipeline with typed result: capture → extract frame height ---
    std::cout << "\nStarting FramePipeline<int>: extract frame height\n";
    FramePipeline<int> height_pipeline(camera, pool);
    height_pipeline.start([](cv::Mat frame){ return frame.rows; });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    if (auto h = height_pipeline.tryPop())
        std::cout << "Frame height: " << *h << " px\n";
    height_pipeline.stop();

    std::cout << "\nDone.\n";
    return 0;
}
