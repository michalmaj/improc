// tests/core/ops/test_remap.cpp
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include "improc/core/ops/remap.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::core;

static std::pair<cv::Mat, cv::Mat> make_identity_maps(int rows, int cols) {
    cv::Mat map1(rows, cols, CV_32FC1);
    cv::Mat map2(rows, cols, CV_32FC1);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            map1.at<float>(r, c) = static_cast<float>(c);
            map2.at<float>(r, c) = static_cast<float>(r);
        }
    return {map1, map2};
}

TEST(RemapTest, ConstructionWithValidMaps) {
    auto [m1, m2] = make_identity_maps(50, 50);
    EXPECT_NO_THROW(Remap(m1, m2));
}

TEST(RemapTest, FluentInterpolationReturnsThis) {
    auto [m1, m2] = make_identity_maps(50, 50);
    Remap op(m1, m2);
    EXPECT_EQ(&op.interpolation(cv::INTER_NEAREST), &op);
}

TEST(RemapTest, IdentityMapsPreserveImage) {
    cv::Mat m(50, 50, CV_8UC3);
    cv::RNG rng(42);
    rng.fill(m, cv::RNG::UNIFORM, 0, 256);
    Image<BGR> img(m.clone());
    auto [map1, map2] = make_identity_maps(50, 50);
    auto result = img | Remap(map1, map2).interpolation(cv::INTER_NEAREST);
    cv::Mat diff;
    cv::absdiff(result.mat(), m, diff);
    EXPECT_EQ(cv::sum(diff)[0] + cv::sum(diff)[1] + cv::sum(diff)[2], 0.0);
}

TEST(RemapTest, EmptyMapThrows) {
    cv::Mat m1(50, 50, CV_32FC1);
    cv::Mat m2;
    EXPECT_THROW(Remap(m1, m2), std::invalid_argument);
}

TEST(RemapTest, MismatchedMapSizesThrow) {
    cv::Mat m1(50, 50, CV_32FC1);
    cv::Mat m2(60, 60, CV_32FC1);
    EXPECT_THROW(Remap(m1, m2), std::invalid_argument);
}

TEST(RemapTest, OutputSizeMatchesMapSize) {
    // Input is 100×100, maps are 50×50 — output should be 50×50
    cv::Mat m(100, 100, CV_8UC1, cv::Scalar(128));
    Image<Gray> img(m);
    auto [map1, map2] = make_identity_maps(50, 50);
    auto result = img | Remap(map1, map2);
    EXPECT_EQ(result.rows(), 50);
    EXPECT_EQ(result.cols(), 50);
}
