// tests/ml/test_classification_eval.cpp
#include <gtest/gtest.h>
#include <vector>
#include "improc/ml/eval/classification.hpp"

using namespace improc::ml;

// ── free functions ────────────────────────────────────────────────────────────

TEST(AccuracyTest, AllCorrect) {
    std::vector<int> p = {0, 1, 2};
    std::vector<int> g = {0, 1, 2};
    EXPECT_FLOAT_EQ(accuracy(p, g), 1.0f);
}

TEST(AccuracyTest, AllWrong) {
    std::vector<int> p = {1, 2, 0};
    std::vector<int> g = {0, 1, 2};
    EXPECT_FLOAT_EQ(accuracy(p, g), 0.0f);
}

TEST(PrecisionTest, BinaryCase) {
    // preds: 0,0,1,1   gts: 0,1,1,1
    // class 1: TP=2, FP=0, col_sum=2 → precision=1.0
    std::vector<int> p = {0, 0, 1, 1};
    std::vector<int> g = {0, 1, 1, 1};
    EXPECT_NEAR(precision_score(p, g, 1), 1.0f, 1e-5f);
}

TEST(RecallTest, BinaryCase) {
    // class 1: TP=2, FN=1, row_sum=3 → recall=2/3
    std::vector<int> p = {0, 0, 1, 1};
    std::vector<int> g = {0, 1, 1, 1};
    EXPECT_NEAR(recall_score(p, g, 1), 2.0f / 3.0f, 1e-5f);
}

TEST(F1Test, BinaryCase) {
    // prec=1.0, rec=2/3 → f1 = 2*(1*2/3)/(1+2/3) = (4/3)/(5/3) = 4/5 = 0.8
    std::vector<int> p = {0, 0, 1, 1};
    std::vector<int> g = {0, 1, 1, 1};
    EXPECT_NEAR(f1_score(p, g, 1), 0.8f, 1e-5f);
}
