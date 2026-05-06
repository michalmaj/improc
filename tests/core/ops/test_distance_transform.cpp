// tests/core/ops/test_distance_transform.cpp
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

TEST(DistanceTransformTest, OutputIsFloat32) {
    Image<Gray> src = make_one_blob();
    Image<Float32> result = src | DistanceTransform{};
    EXPECT_EQ(result.mat().type(), CV_32FC1);
}

TEST(DistanceTransformTest, PreservesSizeFromGray) {
    Image<Gray> src = make_one_blob();
    Image<Float32> result = src | DistanceTransform{};
    EXPECT_EQ(result.rows(), 100);
    EXPECT_EQ(result.cols(), 100);
}

TEST(DistanceTransformTest, ZeroPixelHasZeroDistance) {
    cv::Mat m(50, 50, CV_8UC1, cv::Scalar(255));
    m.at<uchar>(0, 0) = 0;
    Image<Gray> src(m);
    Image<Float32> result = src | DistanceTransform{};
    EXPECT_NEAR(result.mat().at<float>(0, 0), 0.0f, 1e-5f);
}

TEST(DistanceTransformTest, ForegroundPixelHasPositiveDistance) {
    Image<Gray> src = make_one_blob();
    Image<Float32> result = src | DistanceTransform{};
    EXPECT_GT(result.mat().at<float>(49, 49), 0.0f);
}

TEST(DistanceTransformTest, MaskSizeMask5Works) {
    Image<Gray> src = make_one_blob();
    auto result = src | DistanceTransform{}.mask_size(DistanceTransform::MaskSize::Mask5);
    EXPECT_EQ(result.mat().type(), CV_32FC1);
    EXPECT_EQ(result.rows(), 100);
}

TEST(DistanceTransformTest, DistTypeCWorks) {
    Image<Gray> src = make_one_blob();
    auto result = src | DistanceTransform{}.dist_type(DistanceTransform::DistType::C);
    EXPECT_EQ(result.mat().type(), CV_32FC1);
}

TEST(DistanceTransformTest, DistTypeL1VsL2DifferentValues) {
    Image<Gray> src = make_one_blob();
    Image<Float32> l1 = src | DistanceTransform{}.dist_type(DistanceTransform::DistType::L1);
    Image<Float32> l2 = src | DistanceTransform{}.dist_type(DistanceTransform::DistType::L2);
    double diff = cv::norm(l1.mat(), l2.mat(), cv::NORM_L2);
    EXPECT_GT(diff, 0.0);
}
