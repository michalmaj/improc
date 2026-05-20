// tests/core/ops/test_inpaint.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/ops/inpaint.hpp"

using namespace improc::core;

namespace {

Image<BGR> make_random_bgr(int rows = 50, int cols = 50) {
    cv::Mat m(rows, cols, CV_8UC3);
    cv::randu(m, 0, 256);
    return Image<BGR>(m);
}

Image<Gray> make_center_mask(int rows = 50, int cols = 50) {
    cv::Mat m(rows, cols, CV_8UC1, cv::Scalar(0));
    m(cv::Rect(20, 20, 10, 10)).setTo(255);
    return Image<Gray>(m);
}

} // namespace

TEST(InpaintTest, DefaultConstruct) {
    EXPECT_NO_THROW(Inpaint{});
}

TEST(InpaintTest, FluentRadiusReturnsSelf) {
    Inpaint op;
    Inpaint& ref = op.radius(5.0);
    EXPECT_EQ(&ref, &op);
}

TEST(InpaintTest, MaskedRegionChanges) {
    auto img  = make_random_bgr();
    auto mask = make_center_mask();

    auto result = Inpaint{}(img, mask);

    cv::Mat diff;
    cv::absdiff(result.mat()(cv::Rect(20, 20, 10, 10)),
                img.mat()(cv::Rect(20, 20, 10, 10)), diff);
    EXPECT_GT(cv::sum(diff)[0], 0.0);
}

TEST(InpaintTest, OutputSizeMatchesInput) {
    auto img  = make_random_bgr();
    auto mask = make_center_mask();

    auto result = Inpaint{}(img, mask);

    EXPECT_EQ(result.mat().rows, img.mat().rows);
    EXPECT_EQ(result.mat().cols, img.mat().cols);
}

TEST(InpaintTest, MethodNSNoThrow) {
    auto img  = make_random_bgr();
    auto mask = make_center_mask();

    EXPECT_NO_THROW(Inpaint{}.method(InpaintMethod::NS)(img, mask));
}
