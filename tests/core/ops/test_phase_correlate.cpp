// tests/core/ops/test_phase_correlate.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

static Image<Float32> make_float_noise(int rows, int cols) {
    cv::Mat m(rows, cols, CV_32FC1);
    cv::RNG rng(42);
    rng.fill(m, cv::RNG::UNIFORM, 0.f, 1.f);
    return Image<Float32>(m);
}

static Image<Float32> shift_image(const Image<Float32>& img, double dx, double dy) {
    cv::Mat t = (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
    cv::Mat shifted;
    cv::warpAffine(img.mat(), shifted, t, img.mat().size());
    return Image<Float32>(shifted);
}

TEST(PhaseCorrelateTest, DefaultConstruction) {
    EXPECT_NO_THROW(PhaseCorrelate{});
}

TEST(PhaseCorrelateTest, IdenticalImagesGivesNearZeroShift) {
    auto img = make_float_noise(128, 128);
    auto r = PhaseCorrelate{}(img, img);
    EXPECT_NEAR(r.shift.x, 0.0, 0.5);
    EXPECT_NEAR(r.shift.y, 0.0, 0.5);
    EXPECT_GT(r.response, 0.0);
}

TEST(PhaseCorrelateTest, DetectsHorizontalShift) {
    auto prev = make_float_noise(128, 128);
    auto next = shift_image(prev, 5.0, 0.0);
    auto r = PhaseCorrelate{}(prev, next);
    EXPECT_NEAR(r.shift.x, 5.0, 1.0);
    EXPECT_NEAR(r.shift.y, 0.0, 1.0);
}

TEST(PhaseCorrelateTest, MismatchedSizesThrow) {
    Image<Float32> prev(cv::Mat(100, 100, CV_32FC1, cv::Scalar(0.f)));
    Image<Float32> next(cv::Mat(200, 200, CV_32FC1, cv::Scalar(0.f)));
    EXPECT_THROW(PhaseCorrelate{}(prev, next), std::invalid_argument);
}
