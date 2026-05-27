// tests/core/ops/test_quality.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <opencv2/core.hpp>
#include "improc/core/pipeline.hpp"
#include "improc/exceptions.hpp"

using namespace improc::core;

namespace {
Image<BGR> make_bgr_solid(int h, int w, cv::Scalar color = {128, 128, 128}) {
    return Image<BGR>(cv::Mat(h, w, CV_8UC3, color));
}
Image<Gray> make_gray_solid(int h, int w, uchar val = 128) {
    return Image<Gray>(cv::Mat(h, w, CV_8UC1, cv::Scalar(val)));
}
Image<BGR> make_bgr_noise(int h, int w) {
    cv::Mat m(h, w, CV_8UC3);
    cv::randu(m, 0, 255);
    return Image<BGR>(m);
}
} // namespace

// ── PSNR ─────────────────────────────────────────────────────────────────────

TEST(PSNRTest, IdenticalImagesReturnInfinity) {
    auto img = make_bgr_solid(100, 100);
    EXPECT_TRUE(std::isinf(PSNR{}(img, img)));
}

TEST(PSNRTest, DifferentImagesReturnFinitePositive) {
    // black vs gray: PSNR = 10*log10(255²/128²) ≈ 6 dB > 0
    auto ref = make_bgr_solid(100, 100, {0, 0, 0});
    auto cmp = make_bgr_solid(100, 100, {128, 128, 128});
    double val = PSNR{}(ref, cmp);
    EXPECT_GT(val, 0.0);
    EXPECT_FALSE(std::isinf(val));
}

TEST(PSNRTest, WorksOnGray) {
    auto ref = make_gray_solid(100, 100, 0);
    auto cmp = make_gray_solid(100, 100, 128);
    EXPECT_GT(PSNR{}(ref, cmp), 0.0);
}

TEST(PSNRTest, ThrowsOnSizeMismatch) {
    EXPECT_THROW(PSNR{}(make_bgr_solid(100, 100), make_bgr_solid(200, 200)),
                 improc::ParameterError);
}

// ── SSIM ─────────────────────────────────────────────────────────────────────

TEST(SSIMTest, IdenticalImagesReturnOne) {
    auto img = make_bgr_solid(100, 100);
    EXPECT_NEAR(SSIM{}(img, img), 1.0, 0.01);
}

TEST(SSIMTest, DifferentImagesReturnLessThanOne) {
    auto ref = make_bgr_solid(100, 100, {0, 0, 0});
    auto cmp = make_bgr_solid(100, 100, {255, 255, 255});
    EXPECT_LT(SSIM{}(ref, cmp), 1.0);
}

TEST(SSIMTest, WorksOnGray) {
    auto ref = make_gray_solid(100, 100, 50);
    auto cmp = make_gray_solid(100, 100, 200);
    EXPECT_NO_THROW(SSIM{}(ref, cmp));
}

TEST(SSIMTest, ThrowsOnSizeMismatch) {
    EXPECT_THROW(SSIM{}(make_bgr_solid(100, 100), make_bgr_solid(200, 200)),
                 improc::ParameterError);
}

// ── GMSD ─────────────────────────────────────────────────────────────────────

TEST(GMSDTest, IdenticalImagesReturnZero) {
    auto img = make_bgr_solid(100, 100);
    EXPECT_NEAR(GMSD{}(img, img), 0.0, 0.01);
}

TEST(GMSDTest, DifferentImagesReturnNonNegative) {
    EXPECT_GE(GMSD{}(make_bgr_noise(100, 100), make_bgr_noise(100, 100)), 0.0);
}

TEST(GMSDTest, ThrowsOnSizeMismatch) {
    EXPECT_THROW(GMSD{}(make_bgr_solid(100, 100), make_bgr_solid(200, 200)),
                 improc::ParameterError);
}

// ── MSE ──────────────────────────────────────────────────────────────────────

TEST(MSETest, IdenticalImagesReturnZero) {
    auto img = make_bgr_solid(100, 100);
    EXPECT_NEAR(MSE{}(img, img), 0.0, 1e-6);
}

TEST(MSETest, DifferentImagesReturnPositive) {
    auto ref = make_bgr_solid(100, 100, {0, 0, 0});
    auto cmp = make_bgr_solid(100, 100, {255, 255, 255});
    EXPECT_GT(MSE{}(ref, cmp), 0.0);
}

TEST(MSETest, WorksOnGray) {
    auto ref = make_gray_solid(100, 100, 0);
    auto cmp = make_gray_solid(100, 100, 255);
    EXPECT_GT(MSE{}(ref, cmp), 0.0);
}

TEST(MSETest, ThrowsOnSizeMismatch) {
    EXPECT_THROW(MSE{}(make_bgr_solid(100, 100), make_bgr_solid(200, 200)),
                 improc::ParameterError);
}
