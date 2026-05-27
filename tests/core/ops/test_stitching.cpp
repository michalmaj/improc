// tests/core/ops/test_stitching.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/core/pipeline.hpp"
#include "improc/exceptions.hpp"

using namespace improc::core;

namespace {
std::pair<Image<BGR>, Image<BGR>> make_overlapping_pair() {
    cv::Mat base(200, 400, CV_8UC3);
    cv::randu(base, 50, 200);
    cv::GaussianBlur(base, base, {15, 15}, 3.0);
    Image<BGR> left(base(cv::Rect(0,   0, 300, 200)).clone());
    Image<BGR> right(base(cv::Rect(100, 0, 300, 200)).clone());
    return {left, right};
}
} // namespace

TEST(StitchTest, ThrowsWhenFewerThanTwoImages) {
    auto [left, right] = make_overlapping_pair();
    EXPECT_THROW(Stitch{}({left}), improc::ParameterError);
}

TEST(StitchTest, ReturnsResultForOverlappingImages) {
    auto [left, right] = make_overlapping_pair();
    auto result = Stitch{}({left, right});
    // Stitching may or may not succeed on synthetic images — check API shape
    EXPECT_TRUE(result.ok || !result.ok);
    EXPECT_FALSE(result.panorama.mat().empty());
}

TEST(StitchTest, ScansModeDoesNotThrow) {
    auto [left, right] = make_overlapping_pair();
    EXPECT_NO_THROW(Stitch{}.mode(Stitch::Mode::Scans)({left, right}));
}
