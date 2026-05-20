// tests/core/ops/test_grabcut.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/ops/grabcut.hpp"

using namespace improc::core;

namespace {

Image<BGR> make_test_image() {
    cv::Mat m(100, 100, CV_8UC3);
    cv::randu(m, 0, 256);
    return Image<BGR>(m);
}

} // namespace

TEST(GrabCutTest, DefaultConstruct) {
    EXPECT_NO_THROW(GrabCut{});
}

TEST(GrabCutTest, FluentIterationsReturnsSelf) {
    GrabCut op;
    GrabCut& ref = op.iterations(3);
    EXPECT_EQ(&ref, &op);
}

TEST(GrabCutTest, ValidRectReturnsCorrectSize) {
    auto img = make_test_image();
    cv::Rect roi{20, 20, 60, 60};
    GrabCut op;
    op.iterations(2);
    auto result = op(img, roi);
    ASSERT_EQ(result.rows(), 100);
    ASSERT_EQ(result.cols(), 100);
}

TEST(GrabCutTest, EmptyRectThrows) {
    auto img = make_test_image();
    cv::Rect roi{20, 20, 0, 0};
    GrabCut op;
    EXPECT_THROW(op(img, roi), std::invalid_argument);
}

TEST(GrabCutTest, RectOutsideBoundsThrows) {
    auto img = make_test_image();
    cv::Rect roi{50, 50, 80, 80};  // 50+80=130 > 100
    GrabCut op;
    EXPECT_THROW(op(img, roi), std::invalid_argument);
}
