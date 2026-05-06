// tests/core/ops/test_connected_components.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

static Image<Gray> make_one_blob() {
    cv::Mat m(100, 100, CV_8UC1, cv::Scalar(0));
    cv::rectangle(m, {20, 20}, {79, 79}, cv::Scalar(255), -1);
    return Image<Gray>(m);
}

static Image<Gray> make_two_blobs() {
    cv::Mat m(100, 200, CV_8UC1, cv::Scalar(0));
    cv::rectangle(m, {10, 10}, {60, 60}, cv::Scalar(255), -1);
    cv::rectangle(m, {130, 10}, {180, 60}, cv::Scalar(255), -1);
    return Image<Gray>(m);
}

// ── ComponentMap ──────────────────────────────────────────────────────────────

TEST(ComponentMapTest, LabelsMatSameSizeAsSource) {
    Image<Gray> src = make_one_blob();
    ComponentMap result = src | ConnectedComponents{};
    EXPECT_EQ(result.labels.rows, 100);
    EXPECT_EQ(result.labels.cols, 100);
}

TEST(ComponentMapTest, LabelsMatIsCV32S) {
    Image<Gray> src = make_one_blob();
    ComponentMap result = src | ConnectedComponents{};
    EXPECT_EQ(result.labels.type(), CV_32S);
}

TEST(ComponentMapTest, AreaIsPositiveForBlob) {
    Image<Gray> src = make_one_blob();
    ComponentMap result = src | ConnectedComponents{};
    ASSERT_GE(result.count(), 2);
    EXPECT_GT(result.area(1), 0);
}

TEST(ComponentMapTest, BoundingRectIsNonEmpty) {
    Image<Gray> src = make_one_blob();
    ComponentMap result = src | ConnectedComponents{};
    ASSERT_GE(result.count(), 2);
    cv::Rect br = result.bounding_rect(1);
    EXPECT_GT(br.width, 0);
    EXPECT_GT(br.height, 0);
}

TEST(ComponentMapTest, CentroidIsWithinImage) {
    Image<Gray> src = make_one_blob();
    ComponentMap result = src | ConnectedComponents{};
    ASSERT_GE(result.count(), 2);
    cv::Point2d c = result.centroid(1);
    EXPECT_GE(c.x, 0.0);
    EXPECT_GE(c.y, 0.0);
    EXPECT_LT(c.x, 100.0);
    EXPECT_LT(c.y, 100.0);
}

TEST(ComponentMapTest, MaskIsCorrectType) {
    Image<Gray> src = make_one_blob();
    ComponentMap result = src | ConnectedComponents{};
    ASSERT_GE(result.count(), 2);
    cv::Mat m = result.mask(1);
    EXPECT_EQ(m.type(), CV_8U);
    EXPECT_GT(cv::countNonZero(m), 0);
}

TEST(ComponentMapTest, OutOfRangeLabelThrows) {
    Image<Gray> src = make_one_blob();
    ComponentMap result = src | ConnectedComponents{};
    EXPECT_THROW(result.area(result.count()), std::out_of_range);
    EXPECT_THROW(result.area(-1), std::out_of_range);
}

// ── ConnectedComponents ───────────────────────────────────────────────────────

TEST(ConnectedComponentsTest, FindsComponentsInBinaryImage) {
    Image<Gray> src = make_one_blob();
    ComponentMap result = src | ConnectedComponents{};
    EXPECT_GE(result.count(), 2);
}

TEST(ConnectedComponentsTest, BlackImageHasOnlyBackground) {
    Image<Gray> src(cv::Mat(50, 50, CV_8UC1, cv::Scalar(0)));
    ComponentMap result = src | ConnectedComponents{};
    EXPECT_EQ(result.count(), 1);
}

TEST(ConnectedComponentsTest, PipelineSyntax) {
    Image<Gray> src = make_one_blob();
    auto result = src | ConnectedComponents{};
    EXPECT_GE(result.count(), 2);
}

TEST(ConnectedComponentsTest, TwoSeparateBlobsFoundSeparately) {
    Image<Gray> src = make_two_blobs();
    ComponentMap result = src | ConnectedComponents{};
    EXPECT_GE(result.count(), 3);
}

TEST(ConnectedComponentsTest, ConnectivityFourVsEight) {
    cv::Mat m(5, 5, CV_8UC1, cv::Scalar(0));
    m.at<uchar>(0, 0) = 255;
    m.at<uchar>(1, 1) = 255;
    m.at<uchar>(2, 2) = 255;
    Image<Gray> src(m);
    ComponentMap c4 = src | ConnectedComponents{}.connectivity(ConnectedComponents::Connectivity::Four);
    ComponentMap c8 = src | ConnectedComponents{}.connectivity(ConnectedComponents::Connectivity::Eight);
    EXPECT_GT(c4.count(), c8.count());
}
