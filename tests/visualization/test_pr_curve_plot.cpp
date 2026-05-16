// tests/visualization/test_pr_curve_plot.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include "improc/visualization/pr_curve_plot.hpp"

using namespace improc::visualization;

namespace {
using Curves = std::map<std::string,
    std::pair<std::vector<float>, std::vector<float>>>;

Curves one_class_curves() {
    return {{"cat", {{0.0f, 0.5f, 1.0f}, {1.0f, 0.8f, 0.7f}}}};
}
} // namespace

TEST(PRCurvePlotTest, ZeroWidthThrows) {
    EXPECT_THROW(PRCurvePlot{one_class_curves()}.width(0)(),
                 improc::ParameterError);
}
TEST(PRCurvePlotTest, NegativeHeightThrows) {
    EXPECT_THROW(PRCurvePlot{one_class_curves()}.height(-1)(),
                 improc::ParameterError);
}
TEST(PRCurvePlotTest, EmptyCurvesReturnsValidImage) {
    auto img = PRCurvePlot{Curves{}}();
    EXPECT_EQ(img.mat().type(), CV_8UC3);
    EXPECT_GT(img.cols(), 0);
}
TEST(PRCurvePlotTest, OutputSizeMatchesSetters) {
    auto img = PRCurvePlot{one_class_curves()}.width(320).height(240)();
    EXPECT_EQ(img.cols(), 320);
    EXPECT_EQ(img.rows(), 240);
}
TEST(PRCurvePlotTest, DrawingModifiesPixels) {
    auto img = PRCurvePlot{one_class_curves()}();
    cv::Mat gray;
    cv::cvtColor(img.mat(), gray, cv::COLOR_BGR2GRAY);
    EXPECT_GT(cv::countNonZero(gray), 0);
}
TEST(PRCurvePlotTest, RawFormSameOutputSize) {
    auto raw  = one_class_curves();
    std::map<std::string, std::vector<float>> recs, precs;
    for (auto& [k,v] : raw) { recs[k] = v.first; precs[k] = v.second; }

    auto c1 = PRCurvePlot{raw}.width(300)();
    auto c2 = PRCurvePlot{recs, precs}.width(300)();
    EXPECT_EQ(c1.cols(), c2.cols());
}
TEST(PRCurvePlotTest, MAP50BadgeShownWhenSet) {
    // Just verifies it doesn't crash with mAP_50 set
    auto img = PRCurvePlot{one_class_curves()}.mAP_50(0.847f)();
    EXPECT_EQ(img.mat().type(), CV_8UC3);
}
