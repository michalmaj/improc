// tests/visualization/test_class_bar_chart.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include "improc/visualization/class_bar_chart.hpp"
#include "improc/ml/eval/classification.hpp"
#include "improc/ml/eval/detection.hpp"

using namespace improc::visualization;
using improc::ml::ClassMetrics;
using improc::ml::DetectionMetrics;

namespace {
std::map<std::string, std::array<float,3>> sample_raw() {
    return {{"cat", {0.92f, 0.88f, 0.90f}},
            {"dog", {0.85f, 0.87f, 0.86f}}};
}
} // namespace

TEST(ClassBarChartTest, ZeroWidthThrows) {
    EXPECT_THROW(ClassBarChart{sample_raw()}.width(0)(), improc::ParameterError);
}
TEST(ClassBarChartTest, NegativeHeightThrows) {
    EXPECT_THROW(ClassBarChart{sample_raw()}.height(-1)(), improc::ParameterError);
}
TEST(ClassBarChartTest, EmptyDataReturnsValidImage) {
    auto img = ClassBarChart{std::map<std::string, std::array<float,3>>{}}();
    EXPECT_EQ(img.mat().type(), CV_8UC3);
}
TEST(ClassBarChartTest, OutputSizeMatchesSetters) {
    auto img = ClassBarChart{sample_raw()}.width(320).height(240)();
    EXPECT_EQ(img.cols(), 320);
    EXPECT_EQ(img.rows(), 240);
}
TEST(ClassBarChartTest, DrawingModifiesPixels) {
    auto img = ClassBarChart{sample_raw()}();
    cv::Mat gray;
    cv::cvtColor(img.mat(), gray, cv::COLOR_BGR2GRAY);
    EXPECT_GT(cv::countNonZero(gray), 0);
}
TEST(ClassBarChartTest, ClassMetricsStructForm) {
    ClassMetrics m;
    m.per_class_precision = {{"cat", 0.9f}, {"dog", 0.8f}};
    m.per_class_recall    = {{"cat", 0.85f},{"dog", 0.82f}};
    m.per_class_f1        = {{"cat", 0.87f},{"dog", 0.81f}};
    auto img = ClassBarChart{m}.width(300)();
    EXPECT_EQ(img.mat().type(), CV_8UC3);
}
TEST(ClassBarChartTest, DetectionMetricsStructForm) {
    DetectionMetrics m;
    m.per_class_AP = {{"cat", 0.91f}, {"dog", 0.78f}};
    auto img = ClassBarChart{m}.width(300)();
    EXPECT_EQ(img.mat().type(), CV_8UC3);
}
TEST(ClassBarChartTest, RawAndStructSameOutputSize) {
    ClassMetrics m;
    m.per_class_precision = {{"cat", 0.9f}};
    m.per_class_recall    = {{"cat", 0.8f}};
    m.per_class_f1        = {{"cat", 0.85f}};
    auto c1 = ClassBarChart{m}.width(300)();
    auto c2 = ClassBarChart{{{"cat", {0.9f, 0.8f, 0.85f}}}}.width(300)();
    EXPECT_EQ(c1.cols(), c2.cols());
}
