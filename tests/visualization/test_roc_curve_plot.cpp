// tests/visualization/test_roc_curve_plot.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include "improc/visualization/roc_curve_plot.hpp"

using namespace improc::visualization;

namespace {
using RocMap = std::map<std::string,
    std::pair<std::vector<float>, std::vector<float>>>;

RocMap one_class_roc() {
    // fpr: 0.0→1.0, tpr: 0.0→1.0 (perfect classifier)
    return {{"cat", {{0.0f, 0.1f, 0.3f, 1.0f},
                     {0.0f, 0.9f, 1.0f, 1.0f}}}};
}
} // namespace

TEST(ROCCurvePlotTest, ZeroWidthThrows) {
    EXPECT_THROW(ROCCurvePlot{one_class_roc()}.width(0)(),
                 improc::ParameterError);
}
TEST(ROCCurvePlotTest, NegativeHeightThrows) {
    EXPECT_THROW(ROCCurvePlot{one_class_roc()}.height(-1)(),
                 improc::ParameterError);
}
TEST(ROCCurvePlotTest, EmptyInputReturnsValidImage) {
    auto img = ROCCurvePlot{RocMap{}}();
    EXPECT_EQ(img.mat().type(), CV_8UC3);
    EXPECT_GT(img.cols(), 0);
}
TEST(ROCCurvePlotTest, OutputSizeMatchesSetters) {
    auto img = ROCCurvePlot{one_class_roc()}.width(320).height(240)();
    EXPECT_EQ(img.cols(), 320);
    EXPECT_EQ(img.rows(), 240);
}
TEST(ROCCurvePlotTest, DrawingModifiesPixels) {
    auto img = ROCCurvePlot{one_class_roc()}();
    cv::Mat gray;
    cv::cvtColor(img.mat(), gray, cv::COLOR_BGR2GRAY);
    EXPECT_GT(cv::countNonZero(gray), 0);
}
TEST(ROCCurvePlotTest, RawFormTwoConstructors) {
    auto roc = one_class_roc();
    std::map<std::string, std::vector<float>> fprs, tprs;
    for (auto& [k,v] : roc) { fprs[k] = v.first; tprs[k] = v.second; }
    auto c1 = ROCCurvePlot{roc}.width(300)();
    auto c2 = ROCCurvePlot{fprs, tprs}.width(300)();
    EXPECT_EQ(c1.cols(), c2.cols());
}
