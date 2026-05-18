// tests/threading/test_frame_pipeline.cpp
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <thread>
#include "improc/exceptions.hpp"
#include "improc/io/camera_source.hpp"
#include "improc/io/any_camera_source.hpp"
#include "improc/threading/frame_pipeline.hpp"
#include "improc/threading/thread_pool.hpp"

using namespace improc;
using namespace improc::io;
using namespace improc::threading;
using namespace improc::core;

namespace {

struct SpinSource {
    std::atomic<int> count{0};
    bool started = false;

    void start() { started = true; }
    void stop() {}
    std::expected<CameraFrame, Error> getFrame() {
        ++count;
        CameraFrame f;
        f.source_id = "spin";
        f.timestamp = std::chrono::steady_clock::now();
        cv::Mat mat(4, 4, CV_8UC3, cv::Scalar(count.load() % 255));
        f.rgb = Image<BGR>(mat);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return f;
    }
};
static_assert(CameraSourceType<SpinSource>);

} // namespace

TEST(FramePipelineTest, TryPopBeforeStartReturnsNullopt) {
    ThreadPool pool(1);
    SpinSource source;
    FramePipeline<int> pipeline(source, pool);
    EXPECT_FALSE(pipeline.tryPop().has_value());
}

TEST(FramePipelineTest, StartTwiceThrows) {
    ThreadPool pool(1);
    SpinSource source;
    FramePipeline<int> pipeline(source, pool);
    pipeline.start([](CameraFrame) { return 0; });
    EXPECT_THROW(
        pipeline.start([](CameraFrame) { return 0; }),
        improc::Exception
    );
}

TEST(FramePipelineTest, StopIsIdempotent) {
    ThreadPool pool(1);
    SpinSource source;
    FramePipeline<int> pipeline(source, pool);
    pipeline.start([](CameraFrame) { return 0; });
    pipeline.stop();
    EXPECT_NO_THROW(pipeline.stop());
}

TEST(FramePipelineTest, StartCallsCameraStart) {
    ThreadPool pool(1);
    SpinSource source;
    EXPECT_FALSE(source.started);
    FramePipeline<int> pipeline(source, pool);
    pipeline.start([](CameraFrame) { return 0; });
    EXPECT_TRUE(source.started);
    pipeline.stop();
}

TEST(FramePipelineTest, TryPopReturnsProcessedFrame) {
    ThreadPool pool(2);
    SpinSource source;
    FramePipeline<int> pipeline(source, pool);
    pipeline.start([](CameraFrame f) {
        return static_cast<int>(f.rgb->mat().at<cv::Vec3b>(0, 0)[0]);
    });

    int result = -1;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < deadline) {
        if (auto r = pipeline.tryPop()) {
            result = *r;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    pipeline.stop();
    EXPECT_GE(result, 0) << "No frame processed within timeout";
}

TEST(FramePipelineTest, WorksWithAnyCameraSource) {
    ThreadPool pool(1);
    auto src = AnyCameraSource::make<SpinSource>();
    FramePipeline<std::string> pipeline(src, pool);
    pipeline.start([](CameraFrame f) { return f.source_id; });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    pipeline.stop();
    SUCCEED();
}
