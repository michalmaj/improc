// tests/visualization/test_iou_histogram.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include "improc/visualization/iou_histogram.hpp"

using namespace improc::visualization;

namespace {
std::vector<float> sample_ious() {
    return {0.3f, 0.45f, 0.6f, 0.72f, 0.55f, 0.81f, 0.9f, 0.25f, 0.65f, 0.78f};
}
} // namespace

TEST(IoUHistogramTest, ZeroWidthThrows) {
    EXPECT_THROW(IoUHistogram{sample_ious()}.width(0)(), improc::ParameterError);
}
TEST(IoUHistogramTest, NegativeHeightThrows) {
    EXPECT_THROW(IoUHistogram{sample_ious()}.height(-1)(), improc::ParameterError);
}
TEST(IoUHistogramTest, ZeroBinsThrows) {
    EXPECT_THROW(IoUHistogram{sample_ious()}.bins(0)(), improc::ParameterError);
}
TEST(IoUHistogramTest, ThresholdBelowZeroThrows) {
    EXPECT_THROW(IoUHistogram{sample_ious()}.threshold(-0.1f)(),
                 improc::ParameterError);
}
TEST(IoUHistogramTest, ThresholdAboveOneThrows) {
    EXPECT_THROW(IoUHistogram{sample_ious()}.threshold(1.1f)(),
                 improc::ParameterError);
}
TEST(IoUHistogramTest, EmptyInputReturnsValidImage) {
    auto img = IoUHistogram{std::vector<float>{}}();
    EXPECT_EQ(img.mat().type(), CV_8UC3);
    EXPECT_GT(img.cols(), 0);
}
TEST(IoUHistogramTest, OutputSizeMatchesSetters) {
    auto img = IoUHistogram{sample_ious()}.width(400).height(200)();
    EXPECT_EQ(img.cols(), 400);
    EXPECT_EQ(img.rows(), 200);
}
TEST(IoUHistogramTest, DrawingModifiesPixels) {
    auto img = IoUHistogram{sample_ious()}();
    cv::Mat gray;
    cv::cvtColor(img.mat(), gray, cv::COLOR_BGR2GRAY);
    EXPECT_GT(cv::countNonZero(gray), 0);
}
TEST(IoUHistogramTest, ThresholdBoundaryValid) {
    EXPECT_NO_THROW(IoUHistogram{sample_ious()}.threshold(0.0f)());
    EXPECT_NO_THROW(IoUHistogram{sample_ious()}.threshold(1.0f)());
}
