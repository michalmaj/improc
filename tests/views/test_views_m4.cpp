// tests/views/test_views_m4.cpp
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

// ── views::batch ─────────────────────────────────────────────────────────────

TEST(ViewsM4, BatchSplitsEvenly) {
    auto imgs = make_batch(6);
    std::vector<std::vector<Image<BGR>>> batches;
    for (const auto& b : imgs | views::batch(3))
        batches.push_back(b);
    ASSERT_EQ(batches.size(), 2u);
    EXPECT_EQ(batches[0].size(), 3u);
    EXPECT_EQ(batches[1].size(), 3u);
}

TEST(ViewsM4, BatchLastChunkSmaller) {
    auto imgs = make_batch(5);
    std::vector<std::vector<Image<BGR>>> batches;
    for (const auto& b : imgs | views::batch(2))
        batches.push_back(b);
    ASSERT_EQ(batches.size(), 3u);
    EXPECT_EQ(batches[0].size(), 2u);
    EXPECT_EQ(batches[1].size(), 2u);
    EXPECT_EQ(batches[2].size(), 1u);
}

TEST(ViewsM4, BatchLargerThanSource) {
    auto imgs = make_batch(3);
    std::vector<std::vector<Image<BGR>>> batches;
    for (const auto& b : imgs | views::batch(10))
        batches.push_back(b);
    ASSERT_EQ(batches.size(), 1u);
    EXPECT_EQ(batches[0].size(), 3u);
}

TEST(ViewsM4, BatchEmptySource) {
    std::vector<Image<BGR>> empty;
    std::vector<std::vector<Image<BGR>>> batches;
    for (const auto& b : empty | views::batch(4))
        batches.push_back(b);
    EXPECT_TRUE(batches.empty());
}

TEST(ViewsM4, BatchSourceUnchanged) {
    auto imgs = make_batch(4);
    for (const auto& b : imgs | views::batch(2)) (void)b;
    EXPECT_EQ(imgs.size(), 4u);
    EXPECT_EQ(imgs[0].cols(), 64);
}

TEST(ViewsM4, BatchAfterTransform) {
    auto imgs = make_batch(6);
    std::vector<std::vector<Image<BGR>>> batches;
    for (const auto& b : imgs
            | views::transform(Resize{}.width(32).height(32))
            | views::batch(2))
        batches.push_back(b);
    ASSERT_EQ(batches.size(), 3u);
    for (const auto& batch : batches)
        for (const auto& img : batch)
            EXPECT_EQ(img.cols(), 32);
}

TEST(ViewsM4, BatchAfterFilter) {
    auto imgs = make_batch(6);
    for (int i = 0; i < 3; ++i)
        imgs[i] = make_bgr(128, 128);
    // 3 large images → batch(2) → [2, 1]
    std::vector<std::vector<Image<BGR>>> batches;
    for (const auto& b : imgs
            | views::filter([](const Image<BGR>& img) { return img.cols() == 128; })
            | views::batch(2))
        batches.push_back(b);
    ASSERT_EQ(batches.size(), 2u);
    EXPECT_EQ(batches[0].size(), 2u);
    EXPECT_EQ(batches[1].size(), 1u);
}

TEST(ViewsM4, BatchAfterTake) {
    auto imgs = make_batch(10);
    std::vector<std::vector<Image<BGR>>> batches;
    for (const auto& b : imgs | views::take(6) | views::batch(3))
        batches.push_back(b);
    ASSERT_EQ(batches.size(), 2u);
    EXPECT_EQ(batches[0].size(), 3u);
    EXPECT_EQ(batches[1].size(), 3u);
}

TEST(ViewsM4, BatchAfterDrop) {
    auto imgs = make_batch(8);
    std::vector<std::vector<Image<BGR>>> batches;
    for (const auto& b : imgs | views::drop(4) | views::batch(2))
        batches.push_back(b);
    ASSERT_EQ(batches.size(), 2u);
    for (const auto& batch : batches)
        EXPECT_EQ(batch.size(), 2u);
}
