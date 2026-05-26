// benchmarks/calib/bench_calib.cpp
//
// Camera calibration, stereo, epipolar, and ArUco ops — overhead + throughput.
// All inputs generated synthetically; no external files required.
//
// Build: cmake -DIMPROC_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
//              -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="conan_provider.cmake" \
//              -DCONAN_COMMAND=/opt/anaconda3/envs/conan_cv/bin/conan -B build .
//        cmake --build build --target improc_benchmarks --parallel 2
// Run:   ./build/improc_benchmarks \
//          --benchmark_filter="chessboard|calibrate|undistort|solve_pnp|project_points|stereo|fundamental|essential|recover|triangulate|aruco|charuco|generate_aruco"

#include <benchmark/benchmark.h>
#include <array>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include "improc/calib/pipeline.hpp"

using namespace improc::calib;

namespace {

// ── Camera fixtures ───────────────────────────────────────────────────────────

cv::Mat make_K() {
    return (cv::Mat_<double>(3,3) << 800, 0, 320, 0, 800, 240, 0, 0, 1);
}

cv::Mat make_dist() {
    return cv::Mat::zeros(1, 5, CV_64F);
}

// ── Chessboard image fixture ──────────────────────────────────────────────────

// Renders a chessboard with `squares_w` × `squares_h` alternating squares.
// Inner corners = {squares_w-1, squares_h-1}.
// Border of `border` white pixels is added on all sides.
cv::Mat make_chessboard_gray(int squares_w, int squares_h,
                              int cell_px, int border = 20) {
    int h = squares_h * cell_px + 2 * border;
    int w = squares_w * cell_px + 2 * border;
    cv::Mat m(h, w, CV_8UC1, cv::Scalar(255));
    for (int r = 0; r < squares_h; ++r)
        for (int c = 0; c < squares_w; ++c)
            if ((r + c) % 2 == 0)
                m(cv::Rect(border + c * cell_px,
                           border + r * cell_px,
                           cell_px, cell_px)).setTo(0);
    return m;
}

// ── Calibration data fixtures ─────────────────────────────────────────────────

struct CalibData {
    std::vector<std::vector<cv::Point3f>> obj_pts;
    std::vector<std::vector<cv::Point2f>> img_pts;
    cv::Mat K;
    cv::Mat dist;
    cv::Size img_size{640, 480};
};

// Generates n_views synthetic views of a {9,6} chessboard projected via known K.
// Rotations are varied to avoid a degenerate calibration configuration.
CalibData make_calib_data(int n_views) {
    CalibData d;
    d.K    = make_K();
    d.dist = make_dist();

    auto one_pts = make_chessboard_points({9, 6}, 0.03f);

    const std::array<std::array<double, 3>, 12> rots = {{
        {-0.30, -0.20,  0.00}, {-0.20,  0.10,  0.05}, {-0.10, -0.10,  0.00},
        { 0.00,  0.00,  0.00}, { 0.10,  0.20,  0.00}, { 0.20, -0.10,  0.05},
        { 0.30,  0.10,  0.00}, {-0.25,  0.15,  0.00}, { 0.15, -0.20,  0.00},
        {-0.15,  0.00,  0.10}, { 0.05,  0.25,  0.00}, {-0.05, -0.15,  0.05}
    }};

    for (int v = 0; v < n_views; ++v) {
        const auto& r = rots[v % rots.size()];
        cv::Mat rvec = (cv::Mat_<double>(3,1) << r[0], r[1], r[2]);
        cv::Mat tvec = (cv::Mat_<double>(3,1) << r[0]*0.05, r[1]*0.05,
                        0.50 + v * 0.02);
        std::vector<cv::Point2f> ip;
        cv::projectPoints(one_pts, rvec, tvec, d.K, d.dist, ip);
        d.obj_pts.push_back(one_pts);
        d.img_pts.push_back(ip);
    }
    return d;
}

// Two-camera stereo calibration data (same scene, two cameras with baseline).
struct StereoCalibData {
    std::vector<std::vector<cv::Point3f>> obj_pts;
    std::vector<std::vector<cv::Point2f>> img_pts1, img_pts2;
    cv::Mat K1, dist1, K2, dist2;
    cv::Size img_size{640, 480};
};

StereoCalibData make_stereo_calib_data(int n_views) {
    StereoCalibData d;
    d.K1    = make_K();
    d.dist1 = make_dist();
    d.K2    = (cv::Mat_<double>(3,3) << 780, 0, 310, 0, 780, 235, 0, 0, 1);
    d.dist2 = make_dist();

    auto one_pts = make_chessboard_points({9, 6}, 0.03f);
    const cv::Mat baseline = (cv::Mat_<double>(3,1) << -0.10, 0, 0);

    const std::array<std::array<double, 3>, 12> rots = {{
        {-0.30, -0.20,  0.00}, {-0.20,  0.10,  0.05}, {-0.10, -0.10,  0.00},
        { 0.00,  0.00,  0.00}, { 0.10,  0.20,  0.00}, { 0.20, -0.10,  0.05},
        { 0.30,  0.10,  0.00}, {-0.25,  0.15,  0.00}, { 0.15, -0.20,  0.00},
        {-0.15,  0.00,  0.10}, { 0.05,  0.25,  0.00}, {-0.05, -0.15,  0.05}
    }};

    for (int v = 0; v < n_views; ++v) {
        const auto& r = rots[v % rots.size()];
        cv::Mat rvec  = (cv::Mat_<double>(3,1) << r[0], r[1], r[2]);
        cv::Mat tvec1 = (cv::Mat_<double>(3,1) << r[0]*0.05, r[1]*0.05,
                         0.50 + v * 0.02);
        cv::Mat tvec2 = tvec1 + baseline;

        std::vector<cv::Point2f> ip1, ip2;
        cv::projectPoints(one_pts, rvec, tvec1, d.K1, d.dist1, ip1);
        cv::projectPoints(one_pts, rvec, tvec2, d.K2, d.dist2, ip2);
        d.obj_pts.push_back(one_pts);
        d.img_pts1.push_back(ip1);
        d.img_pts2.push_back(ip2);
    }
    return d;
}

// ── Stereo image fixture ──────────────────────────────────────────────────────

std::pair<Image<Gray>, Image<Gray>> make_stereo_pair(int h, int w) {
    cv::Mat base(h, w, CV_8UC1);
    cv::randu(base, 0, 255);
    cv::GaussianBlur(base, base, {5, 5}, 1.5);
    cv::Mat t = (cv::Mat_<double>(2, 3) << 1, 0, 16, 0, 1, 0);
    cv::Mat shifted;
    cv::warpAffine(base, shifted, t, base.size());
    return {Image<Gray>(base), Image<Gray>(shifted)};
}

// ── Point correspondence fixtures ────────────────────────────────────────────

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>
make_point_pairs(int n) {
    std::vector<cv::Point2f> pts1(n), pts2(n);
    for (int i = 0; i < n; ++i) {
        pts1[i] = {50.f + (i * 31 % 540), 50.f + (i * 17 % 380)};
        pts2[i] = {pts1[i].x + (i % 7) - 3.f, pts1[i].y + (i % 5) - 2.f};
    }
    return {pts1, pts2};
}

// 3D→2D correspondences for PnP benchmarks
struct PnPData {
    std::vector<cv::Point3f> obj_pts;
    std::vector<cv::Point2f> img_pts;
};

PnPData make_pnp_data(int n) {
    auto K    = make_K();
    auto dist = make_dist();
    cv::Mat rvec = (cv::Mat_<double>(3,1) << 0.10, 0.20, 0.00);
    cv::Mat tvec = (cv::Mat_<double>(3,1) << 0.00, 0.00, 2.00);

    std::vector<cv::Point3f> obj_pts;
    obj_pts.reserve(n);
    for (int i = 0; i < n; ++i)
        obj_pts.push_back({(i % 4) * 0.05f - 0.075f,
                           (i / 4) * 0.05f - 0.075f, 0.f});

    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(obj_pts, rvec, tvec, K, dist, img_pts);
    return {obj_pts, img_pts};
}

// Projection matrices for TriangulatePoints
struct TriangData {
    cv::Mat P1, P2;
    std::vector<cv::Point2f> pts1, pts2;
};

TriangData make_triangulate_data(int n) {
    auto K    = make_K();
    auto dist = make_dist();
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t1 = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat t2 = (cv::Mat_<double>(3,1) << -0.10, 0, 0);
    cv::Mat Rt1, Rt2;
    cv::hconcat(R, t1, Rt1);
    cv::hconcat(R, t2, Rt2);
    cv::Mat P1 = K * Rt1;
    cv::Mat P2 = K * Rt2;

    std::vector<cv::Point3f> obj_pts;
    obj_pts.reserve(n);
    for (int i = 0; i < n; ++i)
        obj_pts.push_back({(i % 10) * 0.05f - 0.25f,
                           (i / 10) * 0.05f - 0.10f,
                           2.0f + (i % 5) * 0.10f});

    cv::Mat zero = cv::Mat::zeros(3, 1, CV_64F);
    std::vector<cv::Point2f> p1, p2;
    cv::projectPoints(obj_pts, zero, t1, K, dist, p1);
    cv::projectPoints(obj_pts, zero, t2, K, dist, p2);
    return {P1, P2, p1, p2};
}

// ── ArUco fixture ─────────────────────────────────────────────────────────────

cv::Mat make_aruco_scene(const cv::aruco::Dictionary& dict,
                          int id, int marker_px = 200) {
    cv::Mat marker_gray;
    cv::aruco::generateImageMarker(dict, id, marker_px, marker_gray, 1);
    cv::Mat bgr;
    cv::cvtColor(marker_gray, bgr, cv::COLOR_GRAY2BGR);
    cv::Mat scene(400, 400, CV_8UC3, cv::Scalar(255, 255, 255));
    int off = (400 - marker_px) / 2;
    bgr.copyTo(scene(cv::Rect(off, off, marker_px, marker_px)));
    return scene;
}

cv::Mat make_charuco_scene(cv::Size board_size, int square_px, float marker_ratio) {
    auto dict   = ArucoDict{}(cv::aruco::DICT_4X4_50);
    float mk_px = square_px * marker_ratio;
    cv::aruco::CharucoBoard board(board_size,
                                  static_cast<float>(square_px), mk_px, dict);
    int w = board_size.width  * square_px + 2 * square_px;
    int h = board_size.height * square_px + 2 * square_px;
    cv::Mat gray;
    board.generateImage(cv::Size(w, h), gray, square_px, 1);
    cv::Mat result;
    cv::cvtColor(gray, result, cv::COLOR_GRAY2BGR);
    return result;
}

} // namespace

// ── FindChessboardCorners — throughput ───────────────────────────────────────
// SD: 10×7 squares at 60px → 640×460 image; inner corners {9,6}
// HD: 10×7 squares at 90px → 940×670 image; inner corners {9,6}

static void BM_find_chessboard_corners(benchmark::State& state) {
    int cell = state.range(0);
    cv::Mat board = make_chessboard_gray(10, 7, cell);
    Image<Gray> img(board);
    for (auto _ : state)
        benchmark::DoNotOptimize(
            FindChessboardCorners{}.board_size({9, 6})(img));
}
BENCHMARK(BM_find_chessboard_corners)->Arg(60)->Arg(90)->Iterations(5);

// ── FindChessboardCornersSB — throughput ─────────────────────────────────────

static void BM_find_chessboard_corners_sb(benchmark::State& state) {
    int cell = state.range(0);
    cv::Mat board = make_chessboard_gray(10, 7, cell);
    Image<Gray> img(board);
    for (auto _ : state)
        benchmark::DoNotOptimize(
            FindChessboardCornersSB{}.board_size({9, 6})(img));
}
BENCHMARK(BM_find_chessboard_corners_sb)->Arg(60)->Arg(90)->Iterations(5);

// ── RefineCorners — throughput ────────────────────────────────────────────────

static void BM_refine_corners(benchmark::State& state) {
    int cell = state.range(0);
    cv::Mat board = make_chessboard_gray(10, 7, cell);
    Image<Gray> img(board);
    auto result = FindChessboardCorners{}.board_size({9, 6})(img);
    if (!result.found || result.corners.empty()) return;
    for (auto _ : state)
        benchmark::DoNotOptimize(RefineCorners{}(img, result.corners));
}
BENCHMARK(BM_refine_corners)->Arg(60)->Arg(90)->Iterations(5);

// ── CalibrateCamera — one-shot ────────────────────────────────────────────────

static void BM_calibrate_camera(benchmark::State& state) {
    auto d = make_calib_data(10);
    for (auto _ : state)
        benchmark::DoNotOptimize(
            CalibrateCamera{}(d.obj_pts, d.img_pts, d.img_size));
}
BENCHMARK(BM_calibrate_camera)->Iterations(5);

// ── StereoCalibrate — one-shot ────────────────────────────────────────────────

static void BM_stereo_calibrate(benchmark::State& state) {
    auto d = make_stereo_calib_data(10);
    for (auto _ : state)
        benchmark::DoNotOptimize(
            StereoCalibrate{}.K1(d.K1).dist1(d.dist1)
                              .K2(d.K2).dist2(d.dist2)
                              .flags(cv::CALIB_FIX_INTRINSIC)(
                d.obj_pts, d.img_pts1, d.img_pts2, d.img_size));
}
BENCHMARK(BM_stereo_calibrate)->Iterations(5);

// ── Undistort — overhead ──────────────────────────────────────────────────────

static void BM_raw_undistort(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC3);
    cv::randu(src, 0, 255);
    auto K    = make_K();
    auto dist = make_dist();
    for (auto _ : state) {
        cv::Mat dst;
        cv::undistort(src, dst, K, dist);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_raw_undistort)->Args({480, 640})->Args({720, 1280});

static void BM_improc_undistort(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC3);
    cv::randu(src, 0, 255);
    Image<BGR> img(src);
    auto K    = make_K();
    auto dist = make_dist();
    // Undistort defaults match cv::undistort with identity new camera matrix
    for (auto _ : state)
        benchmark::DoNotOptimize(Undistort{}.K(K).dist(dist)(img));
}
BENCHMARK(BM_improc_undistort)->Args({480, 640})->Args({720, 1280});

// ── Undistort — throughput ────────────────────────────────────────────────────

static void BM_undistort(benchmark::State& state) {
    cv::Mat src(state.range(0), state.range(1), CV_8UC3);
    cv::randu(src, 0, 255);
    Image<BGR> img(src);
    auto K    = make_K();
    auto dist = make_dist();
    for (auto _ : state)
        benchmark::DoNotOptimize(Undistort{}.K(K).dist(dist)(img));
}
BENCHMARK(BM_undistort)->Args({480, 640})->Args({720, 1280})->Iterations(5);

// ── UndistortMap — throughput (map init only) ─────────────────────────────────

static void BM_undistort_map(benchmark::State& state) {
    auto K    = make_K();
    auto dist = make_dist();
    cv::Size sz{static_cast<int>(state.range(1)), static_cast<int>(state.range(0))};
    for (auto _ : state)
        benchmark::DoNotOptimize(UndistortMap{}.K(K).dist(dist)(sz));
}
BENCHMARK(BM_undistort_map)->Args({480, 640})->Args({720, 1280})->Iterations(5);
