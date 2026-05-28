// examples/core/demo_camshift.cpp
//
// Demo: CamShift, MeanShift, PhaseCorrelate
//
// Creates a synthetic orange circle on a grey background, generates a second
// frame with the circle shifted by (30, 20) px. Tracks with CamShift
// (rotation-aware rotated rect) and MeanShift (axis-aligned rect), then
// verifies the global frame shift with PhaseCorrelate.
// Press any key to advance.

#include "improc/core/pipeline.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace improc::core;

static void show(const std::string& title, const cv::Mat& m) {
    cv::imshow(title, m);
    cv::waitKey(0);
}

int main() {
    // ── Synthetic scene: orange circle on grey ────────────────────────────────
    const int W = 320, H = 240;
    cv::Mat raw1(H, W, CV_8UC3, cv::Scalar(60, 60, 60));
    cv::circle(raw1, {120, 110}, 40, {30, 120, 220}, -1);  // orange (BGR)
    Image<BGR> bgr1(raw1);

    cv::Mat raw2(H, W, CV_8UC3, cv::Scalar(60, 60, 60));
    cv::circle(raw2, {150, 130}, 40, {30, 120, 220}, -1);  // shifted +30, +20
    Image<BGR> bgr2(raw2);

    show("1 — frame1", bgr1.mat());
    show("2 — frame2 (circle +30px right, +20px down)", bgr2.mat());

    // ── Build hue histogram from the object ROI in frame1 ────────────────────
    Image<HSV> hsv1 = bgr1 | ToHSV{};
    cv::Mat hue1(H, W, CV_8U);
    int from_to[] = {0, 0};
    cv::mixChannels({hsv1.mat()}, {hue1}, from_to, 1);

    cv::Rect roi{95, 85, 55, 55};
    cv::Mat roi_hue = hue1(roi);
    cv::Mat hist;
    int histSize = 32;
    float range[] = {0.f, 180.f};
    const float* histRange = {range};
    cv::calcHist(&roi_hue, 1, nullptr, cv::Mat{}, hist, 1, &histSize, &histRange);
    cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);

    // ── Back-project onto frame2 ───────────────────────────────────────────────
    Image<HSV> hsv2 = bgr2 | ToHSV{};
    cv::Mat hue2(H, W, CV_8U);
    cv::mixChannels({hsv2.mat()}, {hue2}, from_to, 1);
    cv::Mat bp_mat;
    cv::calcBackProject(&hue2, 1, nullptr, hist, bp_mat, &histRange);
    Image<Gray> bp{bp_mat};

    // ── CamShift ──────────────────────────────────────────────────────────────
    std::cout << "\n--- CamShift ---\n";
    cv::Rect cam_win = roi;
    CamShiftResult cam = CamShift{}
        .epsilon(1.0)    // convergence threshold (default: 1.0)
        .max_iter(10)    // max iterations (default: 10)
        (bp, cam_win);

    std::cout << "Tracked centre: ("
              << cam.object.center.x << ", " << cam.object.center.y << ")\n";
    std::cout << "Expected:       (150, 130)\n";
    cv::Mat vis1 = raw2.clone();
    cv::ellipse(vis1, cam.object, {0, 255, 0}, 2);
    show("3 — CamShift result (green ellipse)", vis1);

    // ── MeanShift ─────────────────────────────────────────────────────────────
    std::cout << "\n--- MeanShift ---\n";
    cv::Rect ms_win = roi;
    int iters = MeanShift{}
        .epsilon(1.0)    // convergence threshold (default: 1.0)
        .max_iter(10)    // max iterations (default: 10)
        (bp, ms_win);

    std::cout << "Converged in " << iters << " iterations\n";
    std::cout << "Window centre: (" << (ms_win.x + ms_win.width / 2)
              << ", "               << (ms_win.y + ms_win.height / 2) << ")\n";
    cv::Mat vis2 = raw2.clone();
    cv::rectangle(vis2, ms_win, {0, 0, 255}, 2);
    show("4 — MeanShift result (red rect)", vis2);

    // ── PhaseCorrelate ────────────────────────────────────────────────────────
    std::cout << "\n--- PhaseCorrelate ---\n";
    Image<Float32> f1 = (bgr1 | ToGray{}) | ToFloat32{};
    Image<Float32> f2 = (bgr2 | ToGray{}) | ToFloat32{};

    PhaseCorrelateResult pc = PhaseCorrelate{}(f1, f2);
    std::cout << "Detected shift: (" << pc.shift.x << ", " << pc.shift.y << ")\n";
    std::cout << "Response:       " << pc.response << "  (>0.3 = reliable)\n";
    std::cout << "Expected:       (30, 20)\n";

    std::cout << "Done.\n";
    return 0;
}
