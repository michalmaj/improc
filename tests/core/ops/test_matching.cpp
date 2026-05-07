// tests/core/ops/test_matching.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;
using improc::ParameterError;

static Image<Gray> make_textured() {
    cv::Mat m(200, 200, CV_8UC1, cv::Scalar(128));
    cv::rectangle(m, {10,  10}, {90,  90},  cv::Scalar(255), -1);
    cv::rectangle(m, {110, 10}, {190, 90},  cv::Scalar(0),   -1);
    cv::circle(m, {50,  150}, 40, cv::Scalar(200), -1);
    cv::circle(m, {150, 150}, 40, cv::Scalar(30),  -1);
    return Image<Gray>(m);
}

static DescriptorSet make_orb_desc(Image<Gray> img) {
    KeypointSet kps = img | DetectORB{};
    return img | DescribeORB{kps};
}

static DescriptorSet make_sift_desc(Image<Gray> img) {
    KeypointSet kps = img | DetectSIFT{};
    return img | DescribeSIFT{kps};
}

TEST(MatchSetTest, DefaultIsEmpty) {
    MatchSet ms;
    EXPECT_EQ(ms.size(), 0u);
    EXPECT_TRUE(ms.empty());
    EXPECT_TRUE(ms.matches.empty());
}

TEST(MatchBFTest, MatchesORBDescriptors) {
    Image<Gray> src = make_textured();
    DescriptorSet desc = make_orb_desc(src);
    MatchSet ms = MatchBF{desc, desc}();
    EXPECT_FALSE(ms.empty());
    EXPECT_EQ(ms.size(), ms.matches.size());
}

TEST(MatchBFTest, MatchesSIFTDescriptors) {
    Image<Gray> src = make_textured();
    DescriptorSet desc = make_sift_desc(src);
    MatchSet ms = MatchBF{desc, desc}();
    EXPECT_FALSE(ms.empty());
}

TEST(MatchBFTest, CrossCheckDoesNotExceedNormal) {
    Image<Gray> src = make_textured();
    DescriptorSet desc = make_orb_desc(src);
    MatchSet normal = MatchBF{desc, desc}();
    MatchSet cross  = MatchBF{desc, desc}.cross_check(true)();
    EXPECT_LE(cross.size(), normal.size());
}

TEST(MatchBFTest, NegativeMaxDistanceThrows) {
    Image<Gray> src = make_textured();
    DescriptorSet desc = make_orb_desc(src);
    EXPECT_THROW((MatchBF{desc, desc}.max_distance(-1.0f)), ParameterError);
}

TEST(MatchBFTest, MaxDistanceZeroMeansNoFilter) {
    Image<Gray> src = make_textured();
    DescriptorSet desc = make_orb_desc(src);
    MatchSet unfiltered = MatchBF{desc, desc}();
    MatchSet filtered   = MatchBF{desc, desc}.max_distance(0.0f)();
    EXPECT_EQ(unfiltered.size(), filtered.size());
}

TEST(MatchFlannTest, MatchesSIFTDescriptors) {
    Image<Gray> src = make_textured();
    DescriptorSet desc = make_sift_desc(src);
    MatchSet ms = MatchFlann{desc, desc}();
    EXPECT_FALSE(ms.empty());
}

TEST(MatchFlannTest, BinaryDescriptorsThrowAtCallTime) {
    Image<Gray> src = make_textured();
    DescriptorSet desc = make_orb_desc(src);
    EXPECT_THROW((MatchFlann{desc, desc}()), ParameterError);
}

TEST(MatchFlannTest, RatioThresholdZeroThrows) {
    Image<Gray> src = make_textured();
    DescriptorSet desc = make_sift_desc(src);
    EXPECT_THROW((MatchFlann{desc, desc}.ratio_threshold(0.0f)), ParameterError);
}

TEST(MatchFlannTest, RatioThresholdAboveOneThrows) {
    Image<Gray> src = make_textured();
    DescriptorSet desc = make_sift_desc(src);
    EXPECT_THROW((MatchFlann{desc, desc}.ratio_threshold(1.5f)), ParameterError);
}

TEST(MatchFlannTest, TighterRatioFewerOrEqualMatches) {
    Image<Gray> src = make_textured();
    DescriptorSet desc = make_sift_desc(src);
    MatchSet tight = MatchFlann{desc, desc}.ratio_threshold(0.5f)();
    MatchSet loose = MatchFlann{desc, desc}.ratio_threshold(0.9f)();
    EXPECT_LE(tight.size(), loose.size());
}

TEST(MatchBFTest, EmptyDescriptorsReturnsEmpty) {
    DescriptorSet empty_ds{};
    EXPECT_TRUE((MatchBF{empty_ds, empty_ds}().empty()));
}

TEST(MatchFlannTest, EmptyDescriptorsReturnsEmpty) {
    DescriptorSet empty_ds{};
    EXPECT_TRUE((MatchFlann{empty_ds, empty_ds}().empty()));
}
