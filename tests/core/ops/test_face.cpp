// tests/core/ops/test_face.cpp
#include <gtest/gtest.h>
#include <filesystem>
#include "improc/core/pipeline.hpp"
#include "improc/exceptions.hpp"

using namespace improc::core;

static constexpr const char* kFaceModel = "tests/core/testdata/face_detection_yunet.onnx";
static constexpr const char* kRecogModel = "tests/core/testdata/face_recognition_sface.onnx";

namespace {
Image<BGR> make_blank_bgr_face() {
    return Image<BGR>(cv::Mat(120, 120, CV_8UC3, cv::Scalar(128, 128, 128)));
}
} // namespace

// ── DetectFaceYN error paths ───────────────────────────────────────────────────

TEST(DetectFaceYNTest, ThrowsIfModelNotSet) {
    Image<BGR> img = make_blank_bgr_face();
    DetectFaceYN op;
    EXPECT_THROW(op(img), improc::ParameterError);
}

TEST(DetectFaceYNTest, ThrowsIfModelFileDoesNotExist) {
    Image<BGR> img = make_blank_bgr_face();
    DetectFaceYN op;
    op.model("nonexistent_model.onnx");
    EXPECT_THROW(op(img), improc::FileNotFoundError);
}

// ── DetectFaceYN detection (skipped without model) ───────────────────────────

TEST(DetectFaceYNTest, DetectsNoFaceInBlankImage) {
    if (!std::filesystem::exists(kFaceModel))
        GTEST_SKIP() << "model not present: " << kFaceModel;
    Image<BGR> img = make_blank_bgr_face();
    DetectFaceYN op;
    op.model(kFaceModel);
    auto result = op(img);
    EXPECT_TRUE(result.empty());
}

// ── RecognizeFace error paths ─────────────────────────────────────────────────

TEST(RecognizeFaceTest, ThrowsIfModelNotSet) {
    Image<BGR> img = make_blank_bgr_face();
    RecognizeFace op;
    EXPECT_THROW(op.embed(img), improc::ParameterError);
}

TEST(RecognizeFaceTest, ThrowsIfModelFileDoesNotExist) {
    Image<BGR> img = make_blank_bgr_face();
    RecognizeFace op;
    op.model("nonexistent_recog.onnx");
    EXPECT_THROW(op.embed(img), improc::FileNotFoundError);
}

// ── RecognizeFace::match() — no model needed ──────────────────────────────────

TEST(RecognizeFaceTest, MatchSameEmbeddingIsOne) {
    cv::Mat emb(1, 128, CV_32F);
    cv::randu(emb, 0.0, 1.0);
    FaceEmbedding fe{emb};
    float sim = RecognizeFace::match(fe, fe);
    EXPECT_NEAR(sim, 1.0f, 1e-5f);
}

TEST(RecognizeFaceTest, MatchOppositeEmbeddingIsNegativeOne) {
    cv::Mat emb(1, 128, CV_32F, cv::Scalar(1.0f));
    FaceEmbedding fe_pos{emb.clone()};
    FaceEmbedding fe_neg{-emb.clone()};
    float sim = RecognizeFace::match(fe_pos, fe_neg);
    EXPECT_NEAR(sim, -1.0f, 1e-5f);
}

TEST(RecognizeFaceTest, MatchRangeIsNegOneToOne) {
    cv::Mat a(1, 128, CV_32F), b(1, 128, CV_32F);
    cv::randu(a, -1.0, 1.0);
    cv::randu(b, -1.0, 1.0);
    FaceEmbedding fe_a{a}, fe_b{b};
    float sim = RecognizeFace::match(fe_a, fe_b);
    EXPECT_GE(sim, -1.0f);
    EXPECT_LE(sim, 1.0f);
}

// ── RecognizeFace::embed() — skipped without model ───────────────────────────

TEST(RecognizeFaceTest, EmbedReturnsFaceEmbedding) {
    if (!std::filesystem::exists(kRecogModel))
        GTEST_SKIP() << "model not present: " << kRecogModel;
    Image<BGR> img = make_blank_bgr_face();
    RecognizeFace op;
    op.model(kRecogModel);
    auto emb = op.embed(img);
    EXPECT_FALSE(emb.empty());
    EXPECT_EQ(emb.descriptor.cols, 128);
}

// ── FaceEmbedding unit tests ───────────────────────────────────────────────────

TEST(FaceEmbeddingTest, DefaultConstructedIsEmpty) {
    FaceEmbedding fe;
    EXPECT_TRUE(fe.empty());
    EXPECT_TRUE(fe.descriptor.empty());
}

TEST(FaceEmbeddingTest, CosineSimilarityReturnsZeroForEmpty) {
    FaceEmbedding fe1, fe2;
    EXPECT_FLOAT_EQ(fe1.cosine_similarity(fe2), 0.0f);
}

TEST(FaceEmbeddingTest, CosineSimilarityOfIdenticalVectorsIsOne) {
    cv::Mat v = (cv::Mat_<float>(1, 4) << 1.0f, 0.0f, 0.0f, 0.0f);
    FaceEmbedding fe1{v.clone()};
    FaceEmbedding fe2{v.clone()};
    EXPECT_NEAR(fe1.cosine_similarity(fe2), 1.0f, 1e-5f);
}
