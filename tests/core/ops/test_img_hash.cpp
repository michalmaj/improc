// tests/core/ops/test_img_hash.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

namespace {
Image<BGR> make_bgr(int h = 100, int w = 100) {
    cv::Mat m(h, w, CV_8UC3);
    cv::randu(m, 0, 255);
    return Image<BGR>(m);
}
// Two structurally distinct images: horizontal blue gradient vs vertical red gradient.
// This pair differs in spatial structure, color channel, and texture — safe for all hash types.
// Uniform or complementary images are degenerate for variance/moment-based hashes.
std::pair<Image<BGR>, Image<BGR>> make_distinct_pair() {
    cv::Mat m1(100, 100, CV_8UC3, cv::Scalar(0));
    cv::Mat m2(100, 100, CV_8UC3, cv::Scalar(0));
    for (int r = 0; r < 100; ++r) {
        for (int c = 0; c < 100; ++c) {
            m1.at<cv::Vec3b>(r, c) = {static_cast<uchar>(c * 2.55), 0, 0}; // Blue horizontal gradient
            m2.at<cv::Vec3b>(r, c) = {0, 0, static_cast<uchar>(r * 2.55)}; // Red vertical gradient
        }
    }
    return {Image<BGR>(m1), Image<BGR>(m2)};
}
} // namespace

// Generic test macro — works for both CV_8U (Hamming) and CV_64F (L2) hashes.
#define HASH_TESTS(OpName)                                                       \
TEST(OpName##Test, SameImageSameHash) {                                          \
    auto img = make_bgr();                                                       \
    auto h1 = OpName{}(img);                                                     \
    auto h2 = OpName{}(img);                                                     \
    EXPECT_NEAR(cv::norm(h1, h2), 0.0, 1e-9);                                   \
}                                                                                \
TEST(OpName##Test, SelfDistanceIsZero) {                                         \
    auto h = OpName{}(make_bgr());                                               \
    EXPECT_NEAR(OpName::distance(h, h), 0.0, 1e-9);                             \
}                                                                                \
TEST(OpName##Test, DifferentImagesPositiveDistance) {                            \
    auto [img1, img2] = make_distinct_pair();                                    \
    EXPECT_GT(OpName::distance(OpName{}(img1), OpName{}(img2)), 0.0);           \
}                                                                                \
TEST(OpName##Test, HashIsNotEmpty) {                                             \
    EXPECT_FALSE(OpName{}(make_bgr()).empty());                                  \
}

HASH_TESTS(AverageHash)
HASH_TESTS(PHash)
HASH_TESTS(MarrHildrethHash)
HASH_TESTS(RadialVarianceHash)
HASH_TESTS(ColorMomentHash)
HASH_TESTS(BlockMeanHash)
