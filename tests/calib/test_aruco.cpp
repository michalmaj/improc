// tests/calib/test_aruco.cpp
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include "improc/calib/pipeline.hpp"

using namespace improc::calib;

namespace {

cv::aruco::Dictionary make_dict() {
    return ArucoDict{}(cv::aruco::DICT_4X4_50);
}

cv::Mat make_K() {
    return (cv::Mat_<double>(3,3) << 800, 0, 200, 0, 800, 200, 0, 0, 1);
}

cv::Mat zero_dist() { return cv::Mat::zeros(1, 5, CV_64F); }

// Generates marker id and embeds it centred in a white 400×400 BGR scene.
// Uses GenerateAruco (must be implemented before this helper is useful).
cv::Mat make_marker_scene(const cv::aruco::Dictionary& dict, int id, int marker_px = 200) {
    auto gray = GenerateAruco{}(dict, id, marker_px);
    cv::Mat bgr;
    cv::cvtColor(gray.mat(), bgr, cv::COLOR_GRAY2BGR);
    cv::Mat scene(400, 400, CV_8UC3, cv::Scalar(255, 255, 255));
    int off = (400 - marker_px) / 2;
    bgr.copyTo(scene(cv::Rect(off, off, marker_px, marker_px)));
    return scene;
}

// Renders a ChArUco board to a BGR image using cv::aruco::CharucoBoard directly.
cv::Mat make_charuco_scene(cv::Size board_size, int square_px, float marker_ratio) {
    auto dict   = make_dict();
    float mk_px = square_px * marker_ratio;
    cv::aruco::CharucoBoard board(board_size,
                                  static_cast<float>(square_px), mk_px, dict);
    int w = board_size.width  * square_px + 2 * square_px;
    int h = board_size.height * square_px + 2 * square_px;
    cv::Mat board_gray;
    board.generateImage(cv::Size(w, h), board_gray, square_px, 1);
    cv::Mat result;
    cv::cvtColor(board_gray, result, cv::COLOR_GRAY2BGR);
    return result;
}

} // namespace

// ── ArucoDict ─────────────────────────────────────────────────────────────────

TEST(ArucoDictTest, ReturnsDictWithNonEmptyBytesList) {
    auto dict = ArucoDict{}(cv::aruco::DICT_4X4_50);
    EXPECT_GT(dict.bytesList.rows, 0);
}

// ── GenerateAruco ─────────────────────────────────────────────────────────────

TEST(GenerateArucoTest, OutputIsSquare) {
    auto dict = make_dict();
    auto img = GenerateAruco{}(dict, 0, 100);
    EXPECT_EQ(img.mat().rows, 100);
    EXPECT_EQ(img.mat().cols, 100);
}

TEST(GenerateArucoTest, OutputIsGray) {
    auto dict = make_dict();
    auto img = GenerateAruco{}(dict, 0, 100);
    EXPECT_EQ(img.mat().type(), CV_8UC1);
}

TEST(GenerateArucoTest, ThrowsOnNegativeId) {
    auto dict = make_dict();
    EXPECT_THROW(GenerateAruco{}(dict, -1, 100), std::invalid_argument);
}

TEST(GenerateArucoTest, ThrowsOnZeroSidePixels) {
    auto dict = make_dict();
    EXPECT_THROW(GenerateAruco{}(dict, 0, 0), std::invalid_argument);
}

// ── DetectAruco ───────────────────────────────────────────────────────────────

TEST(DetectArucoTest, DetectsGeneratedMarker) {
    auto dict  = make_dict();
    auto scene = make_marker_scene(dict, 42);
    Image<BGR> img(scene);
    auto result = DetectAruco{}(img, dict);
    ASSERT_EQ(result.ids.size(), 1u);
    EXPECT_EQ(result.ids[0], 42);
}

TEST(DetectArucoTest, ReturnsEmptyOnBlankImage) {
    auto dict = make_dict();
    cv::Mat blank(400, 400, CV_8UC3, cv::Scalar(255, 255, 255));
    Image<BGR> img(blank);
    auto result = DetectAruco{}(img, dict);
    EXPECT_TRUE(result.ids.empty());
    EXPECT_TRUE(result.corners.empty());
}

TEST(DetectArucoTest, WorksOnGrayImage) {
    auto dict  = make_dict();
    auto scene = make_marker_scene(dict, 7);
    cv::Mat gray;
    cv::cvtColor(scene, gray, cv::COLOR_BGR2GRAY);
    Image<Gray> img(gray);
    auto result = DetectAruco{}(img, dict);
    ASSERT_EQ(result.ids.size(), 1u);
    EXPECT_EQ(result.ids[0], 7);
}

// ── DrawAruco ─────────────────────────────────────────────────────────────────

TEST(DrawArucoTest, OutputSameSize) {
    auto dict  = make_dict();
    auto scene = make_marker_scene(dict, 0);
    Image<BGR> img(scene);
    auto result = DetectAruco{}(img, dict);
    auto out = DrawAruco{}(scene, result);
    EXPECT_EQ(out.rows, scene.rows);
    EXPECT_EQ(out.cols, scene.cols);
}

TEST(DrawArucoTest, OutputIsBGR) {
    auto dict  = make_dict();
    auto scene = make_marker_scene(dict, 0);
    Image<BGR> img(scene);
    auto result = DetectAruco{}(img, dict);
    auto out = DrawAruco{}(scene, result);
    EXPECT_EQ(out.type(), CV_8UC3);
}

TEST(DrawArucoTest, AxesOverloadDoesNotThrow) {
    auto dict  = make_dict();
    auto scene = make_marker_scene(dict, 0);
    Image<BGR> img(scene);
    auto result = DetectAruco{}(img, dict);
    auto K     = make_K();
    auto dist  = zero_dist();
    auto poses = ArucoPose{}(result, K, dist, 0.1f);
    EXPECT_NO_THROW(DrawAruco{}(scene, result, poses, K, dist));
}

// ── ArucoPose ─────────────────────────────────────────────────────────────────

TEST(ArucoPoseTest, ReturnsOneResultPerMarker) {
    auto dict         = make_dict();
    auto scene        = make_marker_scene(dict, 5);
    Image<BGR> img(scene);
    auto aruco_result = DetectAruco{}(img, dict);
    auto poses        = ArucoPose{}(aruco_result, make_K(), zero_dist(), 0.1f);
    EXPECT_EQ(poses.size(), aruco_result.ids.size());
}

TEST(ArucoPoseTest, IdMatchesDetection) {
    auto dict         = make_dict();
    auto scene        = make_marker_scene(dict, 5);
    Image<BGR> img(scene);
    auto aruco_result = DetectAruco{}(img, dict);
    ASSERT_EQ(aruco_result.ids.size(), 1u);
    auto poses = ArucoPose{}(aruco_result, make_K(), zero_dist(), 0.1f);
    ASSERT_EQ(poses.size(), 1u);
    EXPECT_EQ(poses[0].id, 5);
}

TEST(ArucoPoseTest, RvecAndTvecNonEmpty) {
    auto dict         = make_dict();
    auto scene        = make_marker_scene(dict, 5);
    Image<BGR> img(scene);
    auto aruco_result = DetectAruco{}(img, dict);
    ASSERT_EQ(aruco_result.ids.size(), 1u);
    auto poses = ArucoPose{}(aruco_result, make_K(), zero_dist(), 0.1f);
    ASSERT_EQ(poses.size(), 1u);
    EXPECT_FALSE(poses[0].rvec.empty());
    EXPECT_FALSE(poses[0].tvec.empty());
}

TEST(ArucoPoseTest, TvecZIsPositive) {
    // 300px marker in 400px scene → frontal view, z > 0
    auto dict         = make_dict();
    auto scene        = make_marker_scene(dict, 5, 300);
    Image<BGR> img(scene);
    auto aruco_result = DetectAruco{}(img, dict);
    ASSERT_EQ(aruco_result.ids.size(), 1u);
    auto poses = ArucoPose{}(aruco_result, make_K(), zero_dist(), 0.1f);
    ASSERT_EQ(poses.size(), 1u);
    EXPECT_GT(poses[0].tvec.at<double>(2), 0.0);
}

TEST(ArucoPoseTest, FrontalMarkerRotationIsAxisAligned) {
    // marker centred in scene: rotation angle ≈ π (marker Y-up vs camera Y-down),
    // with no in-plane or lateral component (ry ≈ 0, rz ≈ 0)
    auto dict         = make_dict();
    auto scene        = make_marker_scene(dict, 5, 300);
    Image<BGR> img(scene);
    auto aruco_result = DetectAruco{}(img, dict);
    ASSERT_EQ(aruco_result.ids.size(), 1u);
    auto poses = ArucoPose{}(aruco_result, make_K(), zero_dist(), 0.1f);
    ASSERT_EQ(poses.size(), 1u);
    EXPECT_NEAR(cv::norm(poses[0].rvec), CV_PI, 0.05) << "rotation angle should be ~π for frontal marker";
    EXPECT_NEAR(poses[0].rvec.at<double>(1), 0.0, 0.05) << "rvec.y should be ~0 for centered marker";
    EXPECT_NEAR(poses[0].rvec.at<double>(2), 0.0, 0.05) << "rvec.z should be ~0 for centered marker";
}
