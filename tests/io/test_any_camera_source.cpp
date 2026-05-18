// tests/io/test_any_camera_source.cpp
#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include "improc/io/any_camera_source.hpp"
#include "improc/io/camera_source.hpp"

using namespace improc::io;
using namespace improc;
using namespace improc::core;

namespace {
struct CountingSource {
    int calls = 0;
    bool started = false;
    void start() { started = true; }
    void stop() {}
    std::expected<CameraFrame, Error> getFrame() {
        ++calls;
        CameraFrame f;
        f.source_id = "counting";
        f.timestamp = std::chrono::steady_clock::now();
        cv::Mat mat(4, 4, CV_8UC3, cv::Scalar(42));
        f.rgb = Image<BGR>(mat);
        return f;
    }
};
static_assert(CameraSourceType<CountingSource>);
}

TEST(AnyCameraSourceTest, DefaultIsEmpty) {
    AnyCameraSource src;
    EXPECT_FALSE(static_cast<bool>(src));
}

TEST(AnyCameraSourceTest, MakeWrapsConcreteType) {
    auto src = AnyCameraSource::make<CountingSource>();
    EXPECT_TRUE(static_cast<bool>(src));
}

TEST(AnyCameraSourceTest, StartAndGetFrame) {
    auto src = AnyCameraSource::make<CountingSource>();
    src.start();
    auto frame = src.getFrame();
    ASSERT_TRUE(frame.has_value());
    EXPECT_EQ(frame->source_id, "counting");
    src.stop();
}

TEST(AnyCameraSourceTest, SatisfiesCameraSourceTypeConcept) {
    static_assert(CameraSourceType<AnyCameraSource>,
                  "AnyCameraSource must satisfy CameraSourceType");
}

TEST(AnyCameraSourceTest, EmptySourceGetFrameReturnsError) {
    AnyCameraSource src;
    auto frame = src.getFrame();
    EXPECT_FALSE(frame.has_value());
    EXPECT_EQ(frame.error().code, improc::Error::Code::CameraUnavailable);
}

TEST(AnyCameraSourceTest, VectorOfSources) {
    std::vector<AnyCameraSource> cams;
    cams.push_back(AnyCameraSource::make<CountingSource>());
    cams.push_back(AnyCameraSource::make<CountingSource>());
    for (auto& cam : cams) cam.start();
    for (auto& cam : cams) {
        auto f = cam.getFrame();
        EXPECT_TRUE(f.has_value());
    }
}
