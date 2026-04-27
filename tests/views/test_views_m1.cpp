// tests/views/test_views_m1.cpp
#include <gtest/gtest.h>
#include "improc/core/pipeline.hpp"
#include "improc/views/views.hpp"

using namespace improc::core;
using namespace improc::views;

// ── helpers ──────────────────────────────────────────────────────────────────

static Image<BGR> make_bgr(int w, int h, cv::Scalar color = {100, 150, 200}) {
    return Image<BGR>(cv::Mat(h, w, CV_8UC3, color));
}

// ── single transform ─────────────────────────────────────────────────────────

TEST(ViewsM1, SingleTransformBuildsView) {
    auto img  = make_bgr(64, 64);
    auto view = img | views::transform(Resize{}.width(32).height(32));
    SUCCEED();
}

TEST(ViewsM1, SingleTransformMaterializesCorrectSize) {
    auto img    = make_bgr(64, 64);
    auto view   = img | views::transform(Resize{}.width(32).height(32));
    Image<BGR> result = view | views::to<Image<BGR>>();
    EXPECT_EQ(result.cols(), 32);
    EXPECT_EQ(result.rows(), 32);
}

TEST(ViewsM1, SourceImageUnchangedAfterEval) {
    auto img  = make_bgr(64, 64);  // color {100, 150, 200}
    auto view = img | views::transform(Resize{}.width(32).height(32));
    [[maybe_unused]] Image<BGR> result = view | views::to<Image<BGR>>();
    EXPECT_EQ(img.cols(), 64);
    EXPECT_EQ(img.rows(), 64);
    // Verify pixel data is intact, not just dimensions
    auto px = img.mat().at<cv::Vec3b>(32, 32);
    EXPECT_EQ(px[0], 100);  // B
    EXPECT_EQ(px[1], 150);  // G
    EXPECT_EQ(px[2], 200);  // R
}

TEST(ViewsM1, EvalTwiceGivesSameShape) {
    auto img  = make_bgr(64, 64);
    auto view = img | views::transform(Resize{}.width(32).height(32));
    Image<BGR> r1 = view | views::to<Image<BGR>>();
    Image<BGR> r2 = view | views::to<Image<BGR>>();
    EXPECT_EQ(r1.cols(), r2.cols());
    EXPECT_EQ(r1.rows(), r2.rows());
}

// ── chained transforms ───────────────────────────────────────────────────────

TEST(ViewsM1, TwoTransformsChained) {
    auto img  = make_bgr(64, 64);
    auto view = img
        | views::transform(Resize{}.width(32).height(32))
        | views::transform(GaussianBlur{}.kernel_size(3));
    Image<BGR> result = view | views::to<Image<BGR>>();
    EXPECT_EQ(result.cols(), 32);
    EXPECT_EQ(result.rows(), 32);
}

TEST(ViewsM1, ThreeTransformsChained) {
    auto img  = make_bgr(128, 128);
    auto view = img
        | views::transform(Resize{}.width(64).height(64))
        | views::transform(GaussianBlur{}.kernel_size(3))
        | views::transform(Resize{}.width(32).height(32));
    Image<BGR> result = view | views::to<Image<BGR>>();
    EXPECT_EQ(result.cols(), 32);
    EXPECT_EQ(result.rows(), 32);
}

TEST(ViewsM1, ViewStoredAsAutoThenMaterialized) {
    auto img = make_bgr(64, 64);
    auto pipeline = img
        | views::transform(Resize{}.width(16).height(16))
        | views::transform(GaussianBlur{}.kernel_size(3));
    Image<BGR> result = pipeline | views::to<Image<BGR>>();
    EXPECT_EQ(result.cols(), 16);
    EXPECT_EQ(result.rows(), 16);
}

TEST(ViewsM1, PixelValuesCorrectAfterOp) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(0, 255, 0));
    Image<BGR> img{mat};
    auto view = img | views::transform(Resize{}.width(8).height(8));
    Image<BGR> result = view | views::to<Image<BGR>>();
    auto px = result.mat().at<cv::Vec3b>(4, 4);
    EXPECT_EQ(px[1], 255);  // G channel
}
