// examples/calib/demo_calibrate.cpp
//
// Demo: FindChessboardCornersSB, CalibrateCamera, Undistort, UndistortMap + Remap
//
// Synthesises 5 chessboard views by warping a generated board image,
// runs calibration, prints K and RMS, shows undistorted result.
// In a real workflow, replace synthetic views with imread<BGR> from disk.
// Press any key to advance.

#include "improc/calib/pipeline.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace improc::calib;
using namespace improc::core;

static void show(const std::string& t, const cv::Mat& m) {
    cv::imshow(t, m); cv::waitKey(0);
}

int main() {
    const cv::Size board{9, 6};
    const float    sq = 0.025f;

    // ── Synthesise a reference chessboard image ───────────────────────────────
    const int cell = 40, pad = cell;
    int W = board.width  * cell + 2 * pad;
    int H = board.height * cell + 2 * pad;
    cv::Mat ref(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
    for (int r = 0; r <= board.height; ++r)
        for (int c = 0; c <= board.width; ++c)
            if ((r + c) % 2 == 0)
                cv::rectangle(ref,
                    {c * cell + pad, r * cell + pad},
                    {(c+1)*cell + pad, (r+1)*cell + pad},
                    {0, 0, 0}, -1);
    Image<BGR> ref_img(ref);

    // ── Collect calibration views via slight warps ────────────────────────────
    std::vector<std::vector<cv::Point3f>> all_obj;
    std::vector<std::vector<cv::Point2f>> all_img;
    Image<BGR> last_view(ref);

    const std::vector<std::pair<double,double>> angles = {
        {0.0, 0.0}, {5.0, 0.0}, {-5.0, 0.0}, {0.0, 5.0}, {3.0, -3.0}
    };
    for (auto [ax, ay] : angles) {
        double rad = ax * CV_PI / 180.0;
        cv::Mat M = (cv::Mat_<double>(2, 3)
            << std::cos(rad), -std::sin(rad), W * 0.02 * ax,
               std::sin(rad),  std::cos(rad), H * 0.02 * ay);
        cv::Mat warped;
        cv::warpAffine(ref, warped, M, ref.size(), cv::INTER_LINEAR,
                       cv::BORDER_CONSTANT, cv::Scalar(200, 200, 200));
        Image<BGR> view(warped);

        // FindChessboardCornersSB: newer method, sub-pixel built-in
        auto res = view | FindChessboardCornersSB{}.board_size(board);
        if (!res.found) continue;

        all_obj.push_back(make_chessboard_points(board, sq));
        all_img.push_back(res.corners);
        last_view = view;
    }
    std::cout << "Collected " << all_obj.size() << " valid calibration views\n";
    if (all_obj.size() < 3) {
        std::cerr << "Too few views for calibration (need >= 3)\n";
        return 1;
    }
    show("1 — Example calibration view", last_view.mat());

    // ── CalibrateCamera ───────────────────────────────────────────────────────
    CalibrationResult cal = CalibrateCamera{}
        .flags(0)    // 0 = estimate all parameters freely
        (all_obj, all_img, ref.size());

    std::cout << "RMS re-projection error: " << cal.rms << " px\n";
    std::cout << "Camera matrix:\n" << cal.camera_matrix << "\n";
    std::cout << "Dist coeffs: " << cal.dist_coeffs.t() << "\n";

    // ── Undistort: single image via pipeline op ───────────────────────────────
    Image<BGR> undist = last_view
        | Undistort{}.K(cal.camera_matrix).dist(cal.dist_coeffs);
    show("2 — Original view", last_view.mat());
    show("3 — Undistorted (Undistort pipeline op)", undist.mat());

    // ── UndistortMap + Remap: preferred for video (maps computed once) ─────────
    UndistortMapResult maps = UndistortMap{}
        .K(cal.camera_matrix)     // 3×3 camera intrinsics
        .dist(cal.dist_coeffs)    // distortion coefficients
        (ref.size());

    Image<BGR> undist2 = last_view | Remap{maps.map1, maps.map2};
    show("4 — Undistorted (UndistortMap + Remap)", undist2.mat());

    std::cout << "Done.\n";
    return 0;
}
