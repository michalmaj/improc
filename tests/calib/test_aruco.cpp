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
