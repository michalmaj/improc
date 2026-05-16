// tests/visualization/test_confusion_matrix_plot.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include "improc/visualization/confusion_matrix_plot.hpp"

using namespace improc::visualization;
using improc::ml::ConfusionMatrix;
using improc::ml::ClassMetrics;

namespace {

ConfusionMatrix make_cm(std::vector<std::vector<int>> data) {
    int n = static_cast<int>(data.size());
    cv::Mat_<int> m(n, n, 0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            m(i, j) = data[i][j];
    return ConfusionMatrix{m, n};
}

} // namespace

TEST(ConfusionMatrixPlotTest, ZeroWidthThrows) {
    EXPECT_THROW(ConfusionMatrixPlot{make_cm({{1}})}.width(0)(),
                 improc::ParameterError);
}
TEST(ConfusionMatrixPlotTest, NegativeHeightThrows) {
    EXPECT_THROW(ConfusionMatrixPlot{make_cm({{1}})}.height(-1)(),
                 improc::ParameterError);
}

TEST(ConfusionMatrixPlotTest, EmptyMatrixReturnsValidImage) {
    ConfusionMatrix empty{cv::Mat_<int>(), 0};
    auto img = ConfusionMatrixPlot{empty}();
    EXPECT_EQ(img.mat().type(), CV_8UC3);
    EXPECT_GT(img.cols(), 0);
    EXPECT_GT(img.rows(), 0);
}

TEST(ConfusionMatrixPlotTest, OutputSizeMatchesSetters) {
    auto img = ConfusionMatrixPlot{make_cm({{5,1},{0,4}})}
               .width(300).height(300)();
    EXPECT_EQ(img.cols(), 300);
    EXPECT_EQ(img.rows(), 300);
}

TEST(ConfusionMatrixPlotTest, DrawingModifiesPixels) {
    auto img = ConfusionMatrixPlot{make_cm({{92,5,3},{8,87,5},{2,4,94}})}
               .width(400).height(400)();
    cv::Mat gray;
    cv::cvtColor(img.mat(), gray, cv::COLOR_BGR2GRAY);
    EXPECT_GT(cv::countNonZero(gray), 0);
}

TEST(ConfusionMatrixPlotTest, StructFormAndRawFormSameSize) {
    auto cm = make_cm({{9,1},{2,8}});
    auto c1 = ConfusionMatrixPlot{cm}.width(200).height(200)();
    auto c2 = ConfusionMatrixPlot{{{9,1},{2,8}}, {"a","b"}}.width(200).height(200)();
    EXPECT_EQ(c1.cols(), c2.cols());
    EXPECT_EQ(c1.rows(), c2.rows());
}

TEST(ConfusionMatrixPlotTest, ClassMetricsStructForm) {
    ClassMetrics m;
    m.confusion_matrix = make_cm({{5,0},{1,4}});
    auto img = ConfusionMatrixPlot{m}.width(200).height(200)();
    EXPECT_EQ(img.mat().type(), CV_8UC3);
}
