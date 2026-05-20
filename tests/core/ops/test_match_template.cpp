// tests/core/ops/test_match_template.cpp
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include "improc/core/ops/match_template.hpp"

using namespace improc::core;

namespace {

// Creates a 32x32 BGR patch with a noisy pattern so its correlation score is
// unique only at the exact embedded position (avoids TM_CCOEFF_NORMED 0/0 artefacts
// that arise with constant-color patches on a constant-color background).
cv::Mat make_patch_mat() {
    cv::Mat patch(32, 32, CV_8UC3);
    cv::randu(patch, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    return patch;
}

Image<BGR> make_image_with_patch(cv::Point patch_origin, const cv::Mat& patch) {
    cv::Mat img_mat(128, 128, CV_8UC3, cv::Scalar(128, 128, 128));
    patch.copyTo(img_mat(cv::Rect(patch_origin.x, patch_origin.y, 32, 32)));
    return Image<BGR>(img_mat);
}

} // namespace

TEST(MatchTemplateTest, DefaultConstruction) {
    EXPECT_NO_THROW(MatchTemplate{});
}

TEST(MatchTemplateTest, FluentSetterReturnsThis) {
    MatchTemplate op;
    EXPECT_EQ(&op.method(cv::TM_CCORR_NORMED), &op);
}

TEST(MatchTemplateTest, FindsEmbeddedPatchLocation) {
    cv::Mat patch = make_patch_mat();
    Image<BGR> img = make_image_with_patch({50, 50}, patch);
    Image<BGR> templ(patch);

    auto [loc, score] = MatchTemplate{}(img, templ);
    EXPECT_NEAR(loc.x, 50, 2);
    EXPECT_NEAR(loc.y, 50, 2);
}

TEST(MatchTemplateTest, ThrowsWhenTemplateWiderThanImage) {
    Image<BGR> img(cv::Mat(64, 64, CV_8UC3, cv::Scalar(0)));
    Image<BGR> templ(cv::Mat(32, 128, CV_8UC3, cv::Scalar(0)));
    EXPECT_THROW(MatchTemplate{}(img, templ), std::invalid_argument);
}

TEST(MatchTemplateTest, ThrowsWhenTemplateTallerThanImage) {
    Image<BGR> img(cv::Mat(64, 64, CV_8UC3, cv::Scalar(0)));
    Image<BGR> templ(cv::Mat(128, 32, CV_8UC3, cv::Scalar(0)));
    EXPECT_THROW(MatchTemplate{}(img, templ), std::invalid_argument);
}

TEST(MatchTemplateTest, ExactMatchScoreNearOne) {
    cv::Mat patch = make_patch_mat();
    Image<BGR> img = make_image_with_patch({50, 50}, patch);
    Image<BGR> templ(patch);

    auto [loc, score] = MatchTemplate{}(img, templ);
    EXPECT_NEAR(score, 1.0, 1e-3);
}
