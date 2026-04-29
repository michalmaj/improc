// tests/views/test_views_m2.cpp
#include <gtest/gtest.h>
#include <vector>
#include "improc/core/pipeline.hpp"
#include "improc/views/views.hpp"

using namespace improc::core;
namespace views = improc::views;

// ── helpers ──────────────────────────────────────────────────────────────────

static Image<BGR> make_bgr(int w, int h, cv::Scalar color = {100, 150, 200}) {
    return Image<BGR>(cv::Mat(h, w, CV_8UC3, color));
}

static std::vector<Image<BGR>> make_batch(int n, int w = 64, int h = 64) {
    std::vector<Image<BGR>> v;
    for (int i = 0; i < n; ++i)
        v.push_back(make_bgr(w, h, cv::Scalar(i * 20, 100, 200)));
    return v;
}

// ── vector | transform ───────────────────────────────────────────────────────

TEST(ViewsM2, VectorTransformMaterializesAllElements) {
    auto imgs = make_batch(5, 64, 64);
    auto result = imgs
        | views::transform(Resize{}.width(32).height(32))
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(result.size(), 5u);
    for (const auto& img : result) {
        EXPECT_EQ(img.cols(), 32);
        EXPECT_EQ(img.rows(), 32);
    }
}

TEST(ViewsM2, VectorTransformSourceUnchanged) {
    auto imgs = make_batch(3, 64, 64);
    [[maybe_unused]] auto result = imgs
        | views::transform(Resize{}.width(32).height(32))
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(imgs.size(), 3u);
    EXPECT_EQ(imgs[0].cols(), 64);
}

TEST(ViewsM2, VectorTransformLazyIteration) {
    auto imgs = make_batch(4, 64, 64);
    auto view = imgs | views::transform(Resize{}.width(16).height(16));
    int count = 0;
    for (const auto& img : view) {
        EXPECT_EQ(img.cols(), 16);
        ++count;
    }
    EXPECT_EQ(count, 4);
}

// ── filter ───────────────────────────────────────────────────────────────────

TEST(ViewsM2, FilterKeepsMatchingElements) {
    auto imgs = make_batch(6, 64, 64);
    for (int i = 0; i < 3; ++i)
        imgs[i] = make_bgr(128, 128);

    auto result = imgs
        | views::filter([](const Image<BGR>& img) { return img.cols() > 64; })
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(result.size(), 3u);
    for (const auto& img : result)
        EXPECT_EQ(img.cols(), 128);
}

TEST(ViewsM2, FilterRejectsAll) {
    auto imgs = make_batch(4, 64, 64);
    auto result = imgs
        | views::filter([](const Image<BGR>&) { return false; })
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_TRUE(result.empty());
}

TEST(ViewsM2, FilterKeepsAll) {
    auto imgs = make_batch(4, 64, 64);
    auto result = imgs
        | views::filter([](const Image<BGR>&) { return true; })
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(result.size(), 4u);
}

// ── take / drop ───────────────────────────────────────────────────────────────

TEST(ViewsM2, TakeFirstN) {
    auto imgs = make_batch(8, 64, 64);
    auto result = imgs
        | views::take(3)
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(result.size(), 3u);
}

TEST(ViewsM2, TakeMoreThanAvailable) {
    auto imgs = make_batch(3, 64, 64);
    auto result = imgs
        | views::take(100)
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(result.size(), 3u);
}

TEST(ViewsM2, DropFirstN) {
    auto imgs = make_batch(6, 64, 64);
    auto result = imgs
        | views::drop(2)
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(result.size(), 4u);
}

TEST(ViewsM2, DropMoreThanAvailable) {
    auto imgs = make_batch(3, 64, 64);
    auto result = imgs
        | views::drop(10)
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_TRUE(result.empty());
}

// ── composition ───────────────────────────────────────────────────────────────

TEST(ViewsM2, TransformThenFilter) {
    auto imgs = make_batch(6, 64, 64);
    auto result = imgs
        | views::transform(Resize{}.width(32).height(32))
        | views::filter([](const Image<BGR>& img) { return img.cols() == 32; })
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(result.size(), 6u);
    for (const auto& img : result)
        EXPECT_EQ(img.cols(), 32);
}

TEST(ViewsM2, DropThenTake) {
    auto imgs = make_batch(10, 64, 64);
    auto result = imgs
        | views::drop(3)
        | views::take(4)
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(result.size(), 4u);
}

TEST(ViewsM2, TransformFilterTakePipeline) {
    auto imgs = make_batch(10, 64, 64);
    for (int i = 0; i < 5; ++i)
        imgs[i] = make_bgr(128, 128);

    auto result = imgs
        | views::transform(Resize{}.width(32).height(32))
        | views::filter([](const Image<BGR>& img) { return img.cols() == 32; })
        | views::take(4)
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(result.size(), 4u);
}

// ── filter/take/drop | transform (previously missing overloads) ───────────────

TEST(ViewsM2, FilterThenTransform) {
    auto imgs = make_batch(6, 64, 64);
    auto result = imgs
        | views::filter([](const Image<BGR>&) { return true; })
        | views::transform(Resize{}.width(32).height(32))
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(result.size(), 6u);
    for (const auto& img : result)
        EXPECT_EQ(img.cols(), 32);
}

TEST(ViewsM2, TakeThenTransform) {
    auto imgs = make_batch(8, 64, 64);
    auto result = imgs
        | views::take(4)
        | views::transform(Resize{}.width(16).height(16))
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(result.size(), 4u);
    for (const auto& img : result)
        EXPECT_EQ(img.cols(), 16);
}

TEST(ViewsM2, DropThenTransform) {
    auto imgs = make_batch(6, 64, 64);
    auto result = imgs
        | views::drop(2)
        | views::transform(Resize{}.width(16).height(16))
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(result.size(), 4u);
    for (const auto& img : result)
        EXPECT_EQ(img.cols(), 16);
}
