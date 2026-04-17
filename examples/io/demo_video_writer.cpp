// examples/io/demo_video_writer.cpp
//
// Demonstrates VideoWriter in two ways:
//   1. Record N frames from the default camera to out_camera.mp4
//   2. Generate a synthetic gradient video (no camera required) to out_synthetic.mp4
//
// Build:  cmake --build cmake-build-debug --target demo_video_writer
// Run:    ./cmake-build-debug/demo_video_writer

#include <iostream>
#include <chrono>
#include <thread>
#include "improc/io/camera_capture.hpp"
#include "improc/io/video_writer.hpp"
#include "improc/core/pipeline.hpp"

using improc::io::CameraCapture;
using improc::io::VideoWriter;
using improc::core::Image;
using improc::core::BGR;

// --- Helper: try to open a camera ----------------------------------------

static bool camera_available() {
    cv::VideoCapture cap(0);
    return cap.isOpened();
}

// --- Demo 1: camera → VideoWriter (pipeline form) -------------------------

static void demo_camera(const std::string& out_path) {
    std::cout << "[demo_camera] recording 5 s to " << out_path << " ...\n";

    CameraCapture camera(0);
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); // warm-up

    VideoWriter writer{out_path};
    writer.fps(25);

    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    int frames = 0;
    while (std::chrono::steady_clock::now() < end) {
        auto result = camera.getFrame();
        if (!result) continue;

        // pipeline: write frame, then display it
        auto img = Image<BGR>(result->clone());
        img = writer(std::move(img));    // write
        cv::imshow("Recording", img.mat());
        if (cv::waitKey(1) == 27) break; // ESC to stop early
        ++frames;
    }

    cv::destroyAllWindows();
    std::cout << "[demo_camera] wrote " << frames << " frames to " << out_path << '\n';
}

// --- Demo 2: synthetic gradient video (no camera) -------------------------

static void demo_synthetic(const std::string& out_path) {
    std::cout << "[demo_synthetic] writing synthetic video to " << out_path << " ...\n";

    constexpr int W = 320, H = 240, FPS = 25, FRAMES = 75;

    VideoWriter writer{out_path};
    writer.fps(FPS).size(W, H);

    for (int i = 0; i < FRAMES; ++i) {
        cv::Mat mat(H, W, CV_8UC3);
        // Animate a horizontal gradient that shifts over time
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                mat.at<cv::Vec3b>(y, x) = cv::Vec3b{
                    static_cast<uchar>((x + i * 3) % 256),
                    static_cast<uchar>((y + i * 2) % 256),
                    static_cast<uchar>((i * 5)     % 256)
                };

        writer(Image<BGR>(mat));
    }

    std::cout << "[demo_synthetic] done. " << FRAMES << " frames written.\n";
}

// --------------------------------------------------------------------------

int main() {
    // Synthetic demo always runs (no hardware required)
    demo_synthetic("out_synthetic.mp4");

    // Camera demo only if hardware is available
    if (camera_available()) {
        demo_camera("out_camera.mp4");
    } else {
        std::cout << "[demo_camera] no camera found, skipping.\n";
    }

    return 0;
}
