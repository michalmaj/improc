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

// ── ClassEval ────────────────────────────────────────────────────────────────

TEST(ClassEvalTest, PerfectPredictions) {
    ClassEval eval;
    eval.class_names({"cat", "dog"});
    eval.update(0, 0);
    eval.update(1, 1);
    auto m = eval.compute();
    EXPECT_FLOAT_EQ(m.accuracy, 1.0f);
}

TEST(ClassEvalTest, ConfusionMatrixShape) {
    ClassEval eval;
    eval.class_names({"a", "b", "c"});
    eval.update(0, 0);
    auto m = eval.compute();
    EXPECT_EQ(m.confusion_matrix.mat.rows, 3);
    EXPECT_EQ(m.confusion_matrix.mat.cols, 3);
    EXPECT_EQ(m.confusion_matrix.num_classes, 3);
}

TEST(ClassEvalTest, ConfusionMatrixValues) {
    // 2-class: GT=[0,0,1,1], pred=[0,1,0,1]
    // mat[0][0]=1, mat[0][1]=1, mat[1][0]=1, mat[1][1]=1
    ClassEval eval;
    eval.class_names({"neg", "pos"});
    eval.update(0, 0);
    eval.update(1, 0);
    eval.update(0, 1);
    eval.update(1, 1);
    auto m = eval.compute();
    EXPECT_EQ(m.confusion_matrix.mat(0, 0), 1);
    EXPECT_EQ(m.confusion_matrix.mat(0, 1), 1);
    EXPECT_EQ(m.confusion_matrix.mat(1, 0), 1);
    EXPECT_EQ(m.confusion_matrix.mat(1, 1), 1);
}

TEST(ClassEvalTest, PerClassMetrics) {
    ClassEval eval;
    eval.class_names({"cat", "dog"});
    // cat: pred 2x correct, 0 wrong; dog: 1 correct, 1 wrong
    eval.update(0, 0); eval.update(0, 0);
    eval.update(1, 1); eval.update(0, 1);
    auto m = eval.compute();
    EXPECT_NEAR(m.per_class_precision.at("cat"), 2.0f/3.0f, 1e-5f);
    EXPECT_FLOAT_EQ(m.per_class_recall.at("cat"), 1.0f);
}

TEST(ClassEvalTest, DivisionByZeroSafe) {
    ClassEval eval;
    eval.class_names({"a", "b"});
    eval.update(0, 0);  // only class 0 ever predicted
    auto m = eval.compute();
    EXPECT_FLOAT_EQ(m.per_class_precision.at("b"), 0.0f);
    EXPECT_FLOAT_EQ(m.per_class_recall.at("b"), 0.0f);
}

TEST(ClassEvalTest, ClassNamesInKeys) {
    ClassEval eval;
    eval.class_names({"cat", "dog"});
    eval.update(0, 0);
    auto m = eval.compute();
    EXPECT_TRUE(m.per_class_f1.count("cat") > 0);
    EXPECT_TRUE(m.per_class_f1.count("dog") > 0);
}

TEST(ClassEvalTest, ResetClearsState) {
    ClassEval eval;
    eval.class_names({"a", "b"});
    eval.update(0, 0);
    eval.reset();
    auto m = eval.compute();
    EXPECT_FLOAT_EQ(m.accuracy, 0.0f);
}
