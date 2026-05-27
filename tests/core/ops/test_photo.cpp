// tests/core/ops/test_photo.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/pipeline.hpp"
#include "improc/exceptions.hpp"

using namespace improc::core;

namespace {
Image<BGR> make_bgr(int h = 100, int w = 100) {
    cv::Mat m(h, w, CV_8UC3);
    cv::randu(m, 0, 255);
    return Image<BGR>(m);
}
Image<Float32C3> make_hdr(int h = 100, int w = 100) {
    cv::Mat m(h, w, CV_32FC3);
    cv::randu(m, 0.f, 1.f);
    return Image<Float32C3>(m);
}
Image<Gray> make_circle_mask(int h = 100, int w = 100) {
    cv::Mat m(h, w, CV_8UC1, cv::Scalar(0));
    cv::circle(m, {w / 2, h / 2}, 30, cv::Scalar(255), -1);
    return Image<Gray>(m);
}
} // namespace

// ── EdgePreservingFilter ──────────────────────────────────────────────────────

TEST(EdgePreservingFilterTest, ReturnsSameSizeBGR) {
    auto img = make_bgr();
    auto result = EdgePreservingFilter{}(img);
    EXPECT_EQ(result.rows(), 100);
    EXPECT_EQ(result.cols(), 100);
}

TEST(EdgePreservingFilterTest, NormConvModeWorks) {
    auto img = make_bgr();
    auto result = EdgePreservingFilter{}.filter(EdgePreservingFilter::Filter::NormConv)(img);
    EXPECT_FALSE(result.mat().empty());
}

// ── DetailEnhance ─────────────────────────────────────────────────────────────

TEST(DetailEnhanceTest, ReturnsSameSizeBGR) {
    auto img = make_bgr();
    auto result = DetailEnhance{}(img);
    EXPECT_EQ(result.rows(), 100);
    EXPECT_EQ(result.cols(), 100);
}

// ── Stylize ───────────────────────────────────────────────────────────────────

TEST(StylizeTest, ReturnsSameSizeBGR) {
    auto img = make_bgr();
    auto result = Stylize{}(img);
    EXPECT_EQ(result.rows(), 100);
    EXPECT_EQ(result.cols(), 100);
}

TEST(StylizeTest, PipelineComposable) {
    auto img = make_bgr();
    auto result = img | Stylize{}.sigma_s(30.f);
    EXPECT_FALSE(result.mat().empty());
}

// ── PencilSketch ──────────────────────────────────────────────────────────────

TEST(PencilSketchTest, GrayResultIsGray) {
    auto img = make_bgr();
    auto result = PencilSketch{}(img);
    EXPECT_EQ(result.gray.mat().type(), CV_8UC1);
    EXPECT_FALSE(result.gray.mat().empty());
}

TEST(PencilSketchTest, ColorResultIsBGR) {
    auto img = make_bgr();
    auto result = PencilSketch{}(img);
    EXPECT_EQ(result.color.mat().type(), CV_8UC3);
    EXPECT_FALSE(result.color.mat().empty());
}

TEST(PencilSketchTest, ResultSameSize) {
    auto img = make_bgr();
    auto result = PencilSketch{}(img);
    EXPECT_EQ(result.gray.rows(), 100);
    EXPECT_EQ(result.color.cols(), 100);
}

// ── SeamlessClone ─────────────────────────────────────────────────────────────

TEST(SeamlessCloneTest, ReturnsSameSizeAsDst) {
    auto src  = make_bgr(100, 100);
    auto dst  = make_bgr(100, 100);
    auto mask = make_circle_mask(100, 100);
    auto result = SeamlessClone{}(src, dst, mask, {50, 50});
    EXPECT_EQ(result.rows(), 100);
    EXPECT_EQ(result.cols(), 100);
}

TEST(SeamlessCloneTest, MixedModeWorks) {
    auto src  = make_bgr(100, 100);
    auto dst  = make_bgr(100, 100);
    auto mask = make_circle_mask(100, 100);
    auto result = SeamlessClone{}.mode(SeamlessClone::Mode::Mixed)(src, dst, mask, {50, 50});
    EXPECT_FALSE(result.mat().empty());
}

TEST(SeamlessCloneTest, ThrowsWhenCenterOutsideDst) {
    auto src  = make_bgr(100, 100);
    auto dst  = make_bgr(100, 100);
    auto mask = make_circle_mask(100, 100);
    EXPECT_THROW(SeamlessClone{}(src, dst, mask, {200, 200}), improc::ParameterError);
}

// ── NLMeansDenoisingMulti ─────────────────────────────────────────────────────

TEST(NLMeansDenoisingMultiTest, DenoisesThreeFrames) {
    std::vector<Image<BGR>> frames;
    for (int i = 0; i < 3; ++i) frames.push_back(make_bgr());
    auto result = NLMeansDenoisingMulti{}(frames);
    EXPECT_EQ(result.rows(), 100);
    EXPECT_EQ(result.cols(), 100);
}

TEST(NLMeansDenoisingMultiTest, ThrowsWhenFewerThanThreeFrames) {
    std::vector<Image<BGR>> frames{make_bgr(), make_bgr()};
    EXPECT_THROW(NLMeansDenoisingMulti{}(frames), improc::ParameterError);
}

TEST(NLMeansDenoisingMultiTest, ThrowsWhenTemporalWindowSizeEven) {
    std::vector<Image<BGR>> frames;
    for (int i = 0; i < 4; ++i) frames.push_back(make_bgr());
    EXPECT_THROW(NLMeansDenoisingMulti{}.temporal_window_size(2)(frames), improc::ParameterError);
}

TEST(NLMeansDenoisingMultiTest, ThrowsWhenTemporalWindowSizeLessThan3) {
    std::vector<Image<BGR>> frames;
    for (int i = 0; i < 3; ++i) frames.push_back(make_bgr());
    EXPECT_THROW(NLMeansDenoisingMulti{}.temporal_window_size(1)(frames), improc::ParameterError);
}

// ── MergeHDR ─────────────────────────────────────────────────────────────────

TEST(MergeHDRTest, MertensReturnsFloat32C3Image) {
    std::vector<Image<BGR>> imgs{make_bgr(), make_bgr(), make_bgr()};
    auto result = MergeHDR{}.method(MergeHDR::Method::Mertens)(imgs);
    EXPECT_EQ(result.mat().type(), CV_32FC3);
    EXPECT_EQ(result.rows(), 100);
}

TEST(MergeHDRTest, DebevecWithTimesReturnsFloat32C3) {
    std::vector<Image<BGR>> imgs{make_bgr(), make_bgr(), make_bgr()};
    std::vector<float> times{1.f/30.f, 1.f/60.f, 1.f/125.f};
    auto result = MergeHDR{}.method(MergeHDR::Method::Debevec)(imgs, times);
    EXPECT_EQ(result.mat().type(), CV_32FC3);
}

TEST(MergeHDRTest, ThrowsWhenDebevecAndTimesMismatch) {
    std::vector<Image<BGR>> imgs{make_bgr(), make_bgr(), make_bgr()};
    std::vector<float> times{1.f, 2.f};  // wrong size
    EXPECT_THROW(MergeHDR{}.method(MergeHDR::Method::Debevec)(imgs, times),
                 improc::ParameterError);
}

TEST(MergeHDRTest, ThrowsWhenEmpty) {
    EXPECT_THROW(MergeHDR{}({}), improc::ParameterError);
}

// ── ToneMap ───────────────────────────────────────────────────────────────────

TEST(ToneMapTest, ReinhardReturnsBGR) {
    auto img = make_hdr();
    auto result = ToneMap{}.algorithm(ToneMap::Algorithm::Reinhard)(img);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
    EXPECT_EQ(result.rows(), 100);
}

TEST(ToneMapTest, DragoReturnsBGR) {
    auto img = make_hdr();
    auto result = ToneMap{}.algorithm(ToneMap::Algorithm::Drago)(img);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(ToneMapTest, LinearAndMantiukWork) {
    auto img = make_hdr();
    EXPECT_NO_THROW(ToneMap{}.algorithm(ToneMap::Algorithm::Linear)(img));
    EXPECT_NO_THROW(ToneMap{}.algorithm(ToneMap::Algorithm::Mantiuk)(img));
}
