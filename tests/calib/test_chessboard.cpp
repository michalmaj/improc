// tests/calib/test_chessboard.cpp
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include "improc/calib/pipeline.hpp"

using namespace improc::calib;
using namespace improc::core;

namespace {
// board_size = inner corners (e.g., {9,6} → 10×7 squares)
Image<Gray> make_chessboard_img(cv::Size board_size, int square_px = 40) {
    int border = square_px;
    int cols = (board_size.width + 1) * square_px + 2 * border;
    int rows = (board_size.height + 1) * square_px + 2 * border;
    cv::Mat mat(rows, cols, CV_8UC1, cv::Scalar(255));
    for (int r = 0; r <= board_size.height; ++r) {
        for (int c = 0; c <= board_size.width; ++c) {
            if ((r + c) % 2 == 0) {
                cv::Rect rect(border + c * square_px, border + r * square_px,
                              square_px, square_px);
                mat(rect).setTo(0);
            }
        }
    }
    return Image<Gray>(mat);
}
} // namespace

TEST(FindChessboardCornersTest, FindsInnerCornersOnSyntheticBoard) {
    const cv::Size board{9, 6};
    auto result = make_chessboard_img(board) | FindChessboardCorners{}.board_size(board);
    EXPECT_TRUE(result.found);
    EXPECT_EQ(static_cast<int>(result.corners.size()), board.width * board.height);
}

TEST(FindChessboardCornersTest, NotFoundOnRandomNoise) {
    cv::Mat noise(200, 200, CV_8UC1);
    cv::randu(noise, 0, 255);
    auto result = Image<Gray>(noise) | FindChessboardCorners{}.board_size({9, 6});
    EXPECT_FALSE(result.found);
    EXPECT_TRUE(result.corners.empty());
}

TEST(FindChessboardCornersTest, WorksOnBGRInput) {
    const cv::Size board{9, 6};
    cv::Mat bgr;
    cv::cvtColor(make_chessboard_img(board).mat(), bgr, cv::COLOR_GRAY2BGR);
    auto result = Image<BGR>(bgr) | FindChessboardCorners{}.board_size(board);
    EXPECT_TRUE(result.found);
}

TEST(FindChessboardCornersTest, ThrowsWhenBoardSizeNotSet) {
    EXPECT_THROW(make_chessboard_img({9, 6}) | FindChessboardCorners{},
                 improc::ParameterError);
}

TEST(FindChessboardCornersTest, FluentBoardSizeReturnsThis) {
    FindChessboardCorners op;
    EXPECT_EQ(&op.board_size({9, 6}), &op);
}

TEST(FindChessboardCornersSBTest, FindsInnerCornersOnSyntheticBoard) {
    const cv::Size board{9, 6};
    auto result = make_chessboard_img(board) | FindChessboardCornersSB{}.board_size(board);
    EXPECT_TRUE(result.found);
    EXPECT_EQ(static_cast<int>(result.corners.size()), board.width * board.height);
}

TEST(FindChessboardCornersSBTest, NotFoundOnRandomNoise) {
    cv::Mat noise(200, 200, CV_8UC1);
    cv::randu(noise, 0, 255);
    auto result = Image<Gray>(noise) | FindChessboardCornersSB{}.board_size({9, 6});
    EXPECT_FALSE(result.found);
    EXPECT_TRUE(result.corners.empty());
}

TEST(FindChessboardCornersSBTest, WorksOnBGRInput) {
    const cv::Size board{9, 6};
    cv::Mat bgr;
    cv::cvtColor(make_chessboard_img(board).mat(), bgr, cv::COLOR_GRAY2BGR);
    auto result = Image<BGR>(bgr) | FindChessboardCornersSB{}.board_size(board);
    EXPECT_TRUE(result.found);
}

TEST(FindChessboardCornersSBTest, ThrowsWhenBoardSizeNotSet) {
    EXPECT_THROW(make_chessboard_img({9, 6}) | FindChessboardCornersSB{},
                 improc::ParameterError);
}

TEST(FindChessboardCornersSBTest, FluentBoardSizeReturnsThis) {
    FindChessboardCornersSB op;
    EXPECT_EQ(&op.board_size({9, 6}), &op);
}

TEST(RefineCornersTest, RefinedCornersNotEmpty) {
    const cv::Size board{9, 6};
    auto img = make_chessboard_img(board);
    auto found = img | FindChessboardCorners{}.board_size(board);
    ASSERT_TRUE(found.found);
    auto refined = RefineCorners{}(img, found.corners);
    EXPECT_EQ(refined.size(), found.corners.size());
}

TEST(RefineCornersTest, RefinedCornersCloseToInitial) {
    const cv::Size board{9, 6};
    int sq = 40, border = sq;
    auto img = make_chessboard_img(board, sq);
    auto found = img | FindChessboardCorners{}.board_size(board);
    ASSERT_TRUE(found.found);

    auto refined = RefineCorners{}.win_size(5)(img, found.corners);
    for (size_t i = 0; i < found.corners.size(); ++i) {
        cv::Point2f d = refined[i] - found.corners[i];
        float dist = std::sqrt(d.x*d.x + d.y*d.y);
        EXPECT_LT(dist, 5.f) << "corner " << i << " moved too far during refinement";
    }
}

TEST(RefineCornersTest, FluentSettersReturnThis) {
    RefineCorners op;
    EXPECT_EQ(&op.win_size(11), &op);
    EXPECT_EQ(&op.zero_zone(-1), &op);
    EXPECT_EQ(&op.max_iter(30), &op);
    EXPECT_EQ(&op.epsilon(0.001), &op);
}
