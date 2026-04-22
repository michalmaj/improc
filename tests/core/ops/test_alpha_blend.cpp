// tests/core/ops/test_alpha_blend.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/pipeline.hpp"
#include "improc/exceptions.hpp"

using namespace improc::core;
using improc::ParameterError;

TEST(AlphaBlendTest, FullyOpaqueOverlayReplacesBackground) {
    cv::Mat bg_mat(4, 4, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat ol_mat(4, 4, CV_8UC4, cv::Scalar(200, 200, 200, 255));
    Image<BGR>  bg(bg_mat);
    Image<BGRA> overlay(ol_mat);

    Image<BGR> result = bg | AlphaBlend{overlay};

    cv::Vec3b px = result.mat().at<cv::Vec3b>(0, 0);
    EXPECT_NEAR(px[0], 200, 2);
    EXPECT_NEAR(px[1], 200, 2);
    EXPECT_NEAR(px[2], 200, 2);
}

TEST(AlphaBlendTest, FullyTransparentOverlayKeepsBackground) {
    cv::Mat bg_mat(4, 4, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat ol_mat(4, 4, CV_8UC4, cv::Scalar(200, 200, 200, 0));
    Image<BGR>  bg(bg_mat);
    Image<BGRA> overlay(ol_mat);

    Image<BGR> result = bg | AlphaBlend{overlay};

    cv::Vec3b px = result.mat().at<cv::Vec3b>(0, 0);
    EXPECT_NEAR(px[0], 100, 2);
    EXPECT_NEAR(px[1], 100, 2);
    EXPECT_NEAR(px[2], 100, 2);
}

TEST(AlphaBlendTest, SemiTransparentBlend) {
    // bg=(0,0,0), overlay=(200,200,200,128)
    // expected ≈ 128/255 * 200 ≈ 100
    cv::Mat bg_mat(4, 4, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat ol_mat(4, 4, CV_8UC4, cv::Scalar(200, 200, 200, 128));
    Image<BGR>  bg(bg_mat);
    Image<BGRA> overlay(ol_mat);

    Image<BGR> result = bg | AlphaBlend{overlay};

    cv::Vec3b px = result.mat().at<cv::Vec3b>(0, 0);
    EXPECT_NEAR(px[0], 100, 3);
    EXPECT_NEAR(px[1], 100, 3);
    EXPECT_NEAR(px[2], 100, 3);
}

TEST(AlphaBlendTest, MismatchedSizesThrow) {
    cv::Mat bg_mat(4, 4, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat ol_mat(2, 2, CV_8UC4, cv::Scalar(200, 200, 200, 255));
    Image<BGR>  bg(bg_mat);
    Image<BGRA> overlay(ol_mat);

    EXPECT_THROW(bg | AlphaBlend{overlay}, ParameterError);
}

TEST(AlphaBlendTest, OutputIsBGR) {
    cv::Mat bg_mat(4, 4, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat ol_mat(4, 4, CV_8UC4, cv::Scalar(200, 200, 200, 128));
    Image<BGR>  bg(bg_mat);
    Image<BGRA> overlay(ol_mat);

    // Assignment to Image<BGR> verifies return type at compile time
    Image<BGR> result = bg | AlphaBlend{overlay};
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(AlphaBlendTest, OutputSizeMatchesBackground) {
    cv::Mat bg_mat(6, 8, CV_8UC3, cv::Scalar(50, 50, 50));
    cv::Mat ol_mat(6, 8, CV_8UC4, cv::Scalar(150, 150, 150, 200));
    Image<BGR>  bg(bg_mat);
    Image<BGRA> overlay(ol_mat);

    Image<BGR> result = bg | AlphaBlend{overlay};
    EXPECT_EQ(result.rows(), bg.rows());
    EXPECT_EQ(result.cols(), bg.cols());
}
