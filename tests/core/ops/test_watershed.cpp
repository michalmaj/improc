// tests/core/ops/test_watershed.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/ops/watershed.hpp"

using namespace improc::core;

namespace {

Image<BGR> make_two_region_image() {
    cv::Mat m(50, 50, CV_8UC3, cv::Scalar(0));
    m(cv::Rect(0, 0, 25, 50)).setTo(cv::Scalar(255, 0, 0));
    m(cv::Rect(25, 0, 25, 50)).setTo(cv::Scalar(0, 0, 255));
    return Image<BGR>(m);
}

cv::Mat make_seeded_markers() {
    cv::Mat markers(50, 50, CV_32SC1, cv::Scalar(0));
    markers(cv::Rect(5, 5, 10, 40)).setTo(1);
    markers(cv::Rect(35, 5, 10, 40)).setTo(2);
    return markers;
}

} // namespace

TEST(WatershedTest, DefaultConstruct) {
    EXPECT_NO_THROW(Watershed{});
}

TEST(WatershedTest, ValidMarkersNoThrow) {
    auto img     = make_two_region_image();
    auto markers = make_seeded_markers();
    EXPECT_NO_THROW(Watershed{}(img, markers));
}

TEST(WatershedTest, WrongTypeMarkersThrows) {
    auto img = make_two_region_image();
    cv::Mat markers(50, 50, CV_8UC1, cv::Scalar(0));
    EXPECT_THROW(Watershed{}(img, markers), improc::ParameterError);
}

TEST(WatershedTest, WrongSizeMarkersThrows) {
    auto img = make_two_region_image();
    cv::Mat markers(30, 30, CV_32SC1, cv::Scalar(0));
    EXPECT_THROW(Watershed{}(img, markers), improc::ParameterError);
}

TEST(WatershedTest, BothLabelsPresentAfterCall) {
    auto img     = make_two_region_image();
    auto markers = make_seeded_markers();

    Watershed{}(img, markers);

    cv::Mat region1 = (markers == 1);
    cv::Mat region2 = (markers == 2);
    EXPECT_GT(cv::countNonZero(region1), 0);
    EXPECT_GT(cv::countNonZero(region2), 0);
}

TEST(WatershedTest, BoundaryPixelsMarkedMinusOne) {
    auto img     = make_two_region_image();
    auto markers = make_seeded_markers();

    Watershed{}(img, markers);

    cv::Mat boundary = (markers == -1);
    EXPECT_GT(cv::countNonZero(boundary), 0);
}
