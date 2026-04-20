// tests/core/ops/test_apply_mask.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/ops/apply_mask.hpp"
#include "improc/core/pipeline.hpp"
#include "improc/exceptions.hpp"

using namespace improc::core;

TEST(ApplyMaskTest, NoMaskSetThrows) {
    Image<BGR> img(cv::Mat(32, 32, CV_8UC3, cv::Scalar(100, 100, 100)));
    EXPECT_THROW(ApplyMask{}(img), improc::ParameterError);
}

TEST(ApplyMaskTest, SizeMismatchThrows) {
    Image<Gray> mask(cv::Mat(16, 16, CV_8UC1, cv::Scalar(255)));
    Image<BGR>  img(cv::Mat(32, 32, CV_8UC3, cv::Scalar(100, 100, 100)));
    EXPECT_THROW(ApplyMask{}.mask(mask)(img), improc::ParameterError);
}

TEST(ApplyMaskTest, AllWhiteMaskPreservesImage) {
    Image<Gray> mask(cv::Mat(32, 32, CV_8UC1, cv::Scalar(255)));
    Image<BGR>  img(cv::Mat(32, 32, CV_8UC3, cv::Scalar(100, 150, 200)));
    auto result = ApplyMask{}.mask(mask)(img);
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 0)[0], 100);
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 0)[1], 150);
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 0)[2], 200);
}

TEST(ApplyMaskTest, AllBlackMaskZerosImage) {
    Image<Gray> mask(cv::Mat(32, 32, CV_8UC1, cv::Scalar(0)));
    Image<BGR>  img(cv::Mat(32, 32, CV_8UC3, cv::Scalar(100, 150, 200)));
    auto result = ApplyMask{}.mask(mask)(img);
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 0)[0], 0);
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 0)[1], 0);
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 0)[2], 0);
}

TEST(ApplyMaskTest, PartialMaskZerosOutsideRegion) {
    cv::Mat mask_mat(32, 32, CV_8UC1, cv::Scalar(0));
    mask_mat(cv::Rect(0, 0, 16, 32)).setTo(255);
    Image<Gray> mask(mask_mat);
    Image<BGR>  img(cv::Mat(32, 32, CV_8UC3, cv::Scalar(100, 100, 100)));
    auto result = ApplyMask{}.mask(mask)(img);
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 0)[0], 100);
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 31)[0], 0);
}

TEST(ApplyMaskTest, PreservesOutputDimensions) {
    Image<Gray> mask(cv::Mat(48, 64, CV_8UC1, cv::Scalar(255)));
    Image<BGR>  img(cv::Mat(48, 64, CV_8UC3, cv::Scalar(50, 50, 50)));
    auto result = ApplyMask{}.mask(mask)(img);
    EXPECT_EQ(result.rows(), 48);
    EXPECT_EQ(result.cols(), 64);
}

TEST(ApplyMaskTest, GrayImageWithMask) {
    Image<Gray> mask(cv::Mat(32, 32, CV_8UC1, cv::Scalar(255)));
    Image<Gray> img(cv::Mat(32, 32, CV_8UC1, cv::Scalar(200)));
    auto result = ApplyMask{}.mask(mask)(img);
    EXPECT_EQ(result.mat().at<uchar>(0, 0), 200);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(ApplyMaskTest, PipelineFormComposes) {
    Image<Gray> mask(cv::Mat(32, 32, CV_8UC1, cv::Scalar(255)));
    Image<BGR>  img(cv::Mat(32, 32, CV_8UC3, cv::Scalar(100, 100, 100)));
    auto result = img | ApplyMask{}.mask(mask);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
}
