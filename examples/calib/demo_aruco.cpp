// examples/calib/demo_aruco.cpp
//
// Demo: ArucoDict, GenerateAruco, DetectAruco, DrawAruco, ArucoPose
//
// Generates a DICT_4X4_50 marker (ID 7), embeds it in a white canvas,
// detects it in the same image (round-trip), draws corners + ID, then
// estimates pose using a synthetic camera matrix.
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
    // ── Dictionary and marker ─────────────────────────────────────────────────
    cv::aruco::Dictionary dict = ArucoDict{}(cv::aruco::DICT_4X4_50);

    Image<Gray> marker = GenerateAruco{}
        .border_bits(1)    // white border width in bits (default: 1)
        (dict, 7, 200);    // marker ID 7, 200×200 px output image

    show("1 — Generated marker (ID 7)", marker.mat());

    // ── Embed marker in a scene ───────────────────────────────────────────────
    cv::Mat scene(400, 400, CV_8UC3, cv::Scalar(240, 240, 240));
    cv::Mat marker_bgr;
    cv::cvtColor(marker.mat(), marker_bgr, cv::COLOR_GRAY2BGR);
    marker_bgr.copyTo(scene(cv::Rect{100, 100, 200, 200}));
    Image<BGR> scene_img(scene);

    show("2 — Scene with embedded marker", scene);

    // ── Detect markers ────────────────────────────────────────────────────────
    ArucoResult res = DetectAruco{}(scene_img, dict);
    std::cout << "Detected " << res.ids.size() << " marker(s)\n";
    for (std::size_t i = 0; i < res.ids.size(); ++i)
        std::cout << "  ID " << res.ids[i] << "\n";

    // DrawAruco overload 1: corner outlines + ID text
    cv::Mat annotated = DrawAruco{}(scene.clone(), res);
    show("3 — Detected marker (corners + ID)", annotated);

    // ── Pose estimation ───────────────────────────────────────────────────────
    // Synthetic K for 400×400 image
    cv::Mat K = (cv::Mat_<double>(3, 3)
        << 400, 0, 200,
             0, 400, 200,
             0, 0, 1);
    cv::Mat dist = cv::Mat::zeros(5, 1, CV_64F);

    if (!res.ids.empty()) {
        std::vector<ArucoPoseResult> poses = ArucoPose{}(res, K, dist, 0.05f);
        for (const auto& p : poses)
            std::cout << "Marker " << p.id
                      << "  rvec=" << p.rvec.t()
                      << "  tvec=" << p.tvec.t() << "\n";

        // DrawAruco overload 2: adds 3-D axis frames per marker
        cv::Mat with_axes = DrawAruco{}
            .axis_length(0.03f)    // axis length in world units (default: 0.05)
            (scene.clone(), res, poses, K, dist);
        show("4 — Marker with 3-D pose axes", with_axes);
    }

    std::cout << "Done.\n";
    return 0;
}
