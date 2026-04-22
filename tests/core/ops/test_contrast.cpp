// tests/core/ops/test_contrast.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/pipeline.hpp"
#include "improc/exceptions.hpp"

using namespace improc::core;
using improc::ParameterError;

// Helper: 10x10 BGR image with varying rows (values 0, 25, 50, ... 225)
static Image<BGR> make_varied_bgr() {
    cv::Mat mat(10, 10, CV_8UC3);
    for (int r = 0; r < 10; r++)
        mat.row(r).setTo(cv::Scalar(r * 25, r * 25, r * 25));
    return Image<BGR>(mat);
}

TEST(ContrastTest, FactorAboveOneIncreasesStdDev) {
    Image<BGR> img = make_varied_bgr();
    cv::Mat mean_orig, stddev_orig;
    cv::meanStdDev(img.mat(), mean_orig, stddev_orig);

    Image<BGR> result = img | Contrast{}.factor(1.5);
    cv::Mat mean_res, stddev_res;
    cv::meanStdDev(result.mat(), mean_res, stddev_res);

    EXPECT_GT(stddev_res.at<double>(0), stddev_orig.at<double>(0));
}

TEST(ContrastTest, FactorBelowOneDecreasesStdDev) {
    Image<BGR> img = make_varied_bgr();
    cv::Mat mean_orig, stddev_orig;
    cv::meanStdDev(img.mat(), mean_orig, stddev_orig);

    Image<BGR> result = img | Contrast{}.factor(0.5);
    cv::Mat mean_res, stddev_res;
    cv::meanStdDev(result.mat(), mean_res, stddev_res);

    EXPECT_LT(stddev_res.at<double>(0), stddev_orig.at<double>(0));
}

TEST(ContrastTest, FactorOneReturnsEqualImage) {
    cv::Mat mat(50, 50, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    double original_mean = cv::mean(img.mat())[0];
    Image<BGR> result = img | Contrast{}.factor(1.0);
    double result_mean = cv::mean(result.mat())[0];
    EXPECT_NEAR(result_mean, original_mean, 1e-5);
}

TEST(ContrastTest, FactorZeroThrowsParameterError) {
    EXPECT_THROW(Contrast{}.factor(0.0), ParameterError);
}

TEST(ContrastTest, NegativeFactorThrowsParameterError) {
    EXPECT_THROW(Contrast{}.factor(-1.0), ParameterError);
}

TEST(ContrastTest, ClippingHighValuePixels) {
    cv::Mat mat(1, 1, CV_8UC3, cv::Scalar(200, 200, 200));
    Image<BGR> img(mat);
    Image<BGR> result = img | Contrast{}.factor(2.0);
    cv::Vec3b px = result.mat().at<cv::Vec3b>(0, 0);
    EXPECT_EQ(px[0], 255);
    EXPECT_EQ(px[1], 255);
    EXPECT_EQ(px[2], 255);
}

TEST(ContrastTest, WorksOnGray) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    Image<Gray> result = img | Contrast{}.factor(1.5);
    double result_mean = cv::mean(result.mat())[0];
    EXPECT_GT(result_mean, 100.0);
}

TEST(ContrastTest, WorksOnBGR) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(50, 50, 50));
    Image<BGR> img(mat);
    Image<BGR> result = img | Contrast{}.factor(2.0);
    double result_mean = cv::mean(result.mat())[0];
    EXPECT_NEAR(result_mean, 100.0, 1.0);
}

TEST(ContrastTest, DefaultFactorIsOne) {
    cv::Mat mat(50, 50, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    double original_mean = cv::mean(img.mat())[0];
    Image<BGR> result = Contrast{}(img);
    double result_mean = cv::mean(result.mat())[0];
    EXPECT_NEAR(result_mean, original_mean, 1e-5);
}
