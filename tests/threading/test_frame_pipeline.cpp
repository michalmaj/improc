// tests/threading/test_frame_pipeline.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include <chrono>
#include <thread>
#include <opencv2/videoio.hpp>
#include "improc/threading/frame_pipeline.hpp"
#include "improc/threading/thread_pool.hpp"
#include "improc/io/camera_capture.hpp"

using improc::threading::FramePipeline;
using improc::threading::ThreadPool;
using improc::io::CameraCapture;

static bool camera_available() {
    cv::VideoCapture cap(0);
    return cap.isOpened();
}

TEST(FramePipelineTest, TryPopBeforeStartReturnsNullopt) {
    if (!camera_available()) GTEST_SKIP() << "No camera available";
    CameraCapture camera(0);
    ThreadPool pool(2);
    FramePipeline<cv::Mat> pipeline(camera, pool);
    EXPECT_FALSE(pipeline.tryPop().has_value());
}

TEST(FramePipelineTest, StartTwiceThrows) {
    if (!camera_available()) GTEST_SKIP() << "No camera available";
    CameraCapture camera(0);
    ThreadPool pool(2);
    FramePipeline<cv::Mat> pipeline(camera, pool);
    pipeline.start([](cv::Mat frame){ return frame; });
    EXPECT_THROW(
        pipeline.start([](cv::Mat frame){ return frame; }),
        improc::Exception
    );
}

TEST(FramePipelineTest, StopIsIdempotent) {
    if (!camera_available()) GTEST_SKIP() << "No camera available";
    CameraCapture camera(0);
    ThreadPool pool(2);
    FramePipeline<cv::Mat> pipeline(camera, pool);
    pipeline.start([](cv::Mat frame){ return frame; });
    pipeline.stop();
    EXPECT_NO_THROW(pipeline.stop());
}

TEST(FramePipelineTest, TryPopReturnsProcessedFrame) {
    if (!camera_available()) GTEST_SKIP() << "No camera available";
    CameraCapture camera(0);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));  // camera warmup
    ThreadPool pool(2);
    FramePipeline<cv::Mat> pipeline(camera, pool);
    pipeline.start([](cv::Mat frame){ return frame; });

    cv::Mat result;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < deadline) {
        if (auto r = pipeline.tryPop()) {
            result = *r;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_FALSE(result.empty()) << "No frame processed within timeout";
}

TEST(FramePipelineTest, ProcessorTransformsFrame) {
    if (!camera_available()) GTEST_SKIP() << "No camera available";
    CameraCapture camera(0);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    ThreadPool pool(2);
    FramePipeline<int> pipeline(camera, pool);
    pipeline.start([](cv::Mat frame){ return frame.rows; });

    int rows = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < deadline) {
        if (auto r = pipeline.tryPop()) {
            rows = *r;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_GT(rows, 0) << "Processor did not return frame height";
}
