// tests/io/test_oak_d_capture.cpp
//
// Unit tests (CI-safe): verify the stub compiles and returns the correct error
// when OAK-D support is not compiled in.
//
// Integration tests (local only, hardware required):
//   Run with: ./build/improc_tests --gtest_filter="*OakD*"
//   Requires: OAK-D camera connected via USB3, depthai-core enabled:
//     cmake -DIMPROC_WITH_DEPTHAI=ON ...
//
#include "improc/io/oak_d_capture.hpp"
#include <gtest/gtest.h>

namespace improc::io {

// ── Unit tests (always run, no hardware) ────────────────────────────────────

TEST(OakDCaptureTest, SatisfiesCameraSourceTypeConcept) {
    static_assert(CameraSourceType<OakDCapture>);
}

TEST(OakDCaptureTest, StubGetFrameReturnsUnavailableError) {
#ifndef IMPROC_WITH_DEPTHAI
    OakDCapture oak;
    auto result = oak.getFrame();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, improc::Error::Code::CameraUnavailable);
#else
    GTEST_SKIP() << "Stub test not applicable when depthai is enabled";
#endif
}

TEST(OakDCaptureTest, StubStartThrows) {
#ifndef IMPROC_WITH_DEPTHAI
    OakDCapture oak;
    EXPECT_THROW(oak.start(), std::runtime_error);
#else
    GTEST_SKIP() << "Stub test not applicable when depthai is enabled";
#endif
}

TEST(OakDCaptureTest, StubWithMxIdConstructsAndGetFrameReturnsError) {
#ifndef IMPROC_WITH_DEPTHAI
    OakDCapture oak("14442C10D13B8D1300");
    auto result = oak.getFrame();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, improc::Error::Code::CameraUnavailable);
#else
    GTEST_SKIP() << "Stub test not applicable when depthai is enabled";
#endif
}

// ── Integration tests (local only, require OAK-D hardware + depthai ON) ─────

#ifdef IMPROC_WITH_DEPTHAI

static bool oak_d_available() {
    try {
        dai::Device dev;
        return true;
    } catch (...) {
        return false;
    }
}

TEST(OakDCaptureIntegrationTest, StartStopDoesNotCrash) {
    if (!oak_d_available()) GTEST_SKIP() << "No OAK-D device available.";
    OakDCapture oak;
    ASSERT_NO_THROW(oak.start());
    oak.stop();
}

TEST(OakDCaptureIntegrationTest, GetFrameReturnsRgbAndDepth) {
    if (!oak_d_available()) GTEST_SKIP() << "No OAK-D device available.";
    OakDCapture oak;
    oak.start();
    auto result = oak.getFrame();
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_TRUE(result->rgb.has_value());
    EXPECT_TRUE(result->depth.has_value());
    EXPECT_FALSE(result->source_id.empty());
    oak.stop();
}

TEST(OakDCaptureIntegrationTest, GetFrameNotStartedReturnsError) {
    if (!oak_d_available()) GTEST_SKIP() << "No OAK-D device available.";
    OakDCapture oak;
    auto result = oak.getFrame();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, improc::Error::Code::CameraUnavailable);
}

#endif  // IMPROC_WITH_DEPTHAI

}  // namespace improc::io
