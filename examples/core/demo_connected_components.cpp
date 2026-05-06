// examples/core/demo_connected_components.cpp
//
// Demo: ConnectedComponents, ComponentMap, DistanceTransform
//
// Usage: run from the build directory; press any key to advance.

#include "improc/core/pipeline.hpp"
#include <opencv2/highgui.hpp>
#include <format>
#include <iostream>

using namespace improc::core;

static void show(const std::string& title, const cv::Mat& mat) {
    cv::imshow(title, mat);
    cv::waitKey(0);
}

int main() {
    // ── Source: three shapes on dark background ──────────────────────────────
    cv::Mat raw(300, 400, CV_8UC3, cv::Scalar(30, 30, 30));
    cv::rectangle(raw, {30,  30},  {130, 130}, cv::Scalar(220, 220, 220), -1);
    cv::circle(raw,   {250, 80},  55,          cv::Scalar(200, 200, 200), -1);
    cv::ellipse(raw,  {180, 230}, {70, 40}, 0, 0, 360, cv::Scalar(180, 180, 180), -1);
    Image<BGR>  bgr(raw);
    Image<Gray> gray = bgr | ToGray{};

    show("0a — Source BGR",  bgr.mat());

    // ── Binary image ─────────────────────────────────────────────────────────
    Image<Gray> binary = gray | Threshold{}.value(100).mode(ThresholdMode::Binary);
    show("1 — Binary (threshold=100)", binary.mat());

    // ── ConnectedComponents: Eight (default) ─────────────────────────────────
    ComponentMap cm = binary | ConnectedComponents{};
    std::cout << std::format("Labels (Eight): {}\n", cm.count());
    for (int i = 1; i < cm.count(); ++i) {
        auto br  = cm.bounding_rect(i);
        auto cen = cm.centroid(i);
        std::cout << std::format("  [{}] area={}  bbox={}x{}  center=({:.1f},{:.1f})\n",
            i, cm.area(i), br.width, br.height, cen.x, cen.y);
    }

    // ── Colorise label image ──────────────────────────────────────────────────
    cv::Mat colored(cm.labels.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<cv::Scalar> palette = {
        {0,0,0}, {0,255,0}, {0,0,255}, {255,0,0}, {0,255,255}, {255,0,255}
    };
    for (int i = 0; i < cm.labels.rows; ++i)
        for (int j = 0; j < cm.labels.cols; ++j) {
            int lbl = cm.labels.at<int>(i, j);
            if (lbl > 0 && lbl < static_cast<int>(palette.size()))
                colored.at<cv::Vec3b>(i, j) = cv::Vec3b(
                    static_cast<uchar>(palette[lbl][0]),
                    static_cast<uchar>(palette[lbl][1]),
                    static_cast<uchar>(palette[lbl][2]));
        }
    show("2 — Label map (colorised)", colored);

    // ── Mask for a single component ───────────────────────────────────────────
    if (cm.count() >= 2) {
        show("3 — Mask for label 1", cm.mask(1));
    }

    // ── Connectivity::Four vs Eight ───────────────────────────────────────────
    ComponentMap cm4 = binary | ConnectedComponents{}.connectivity(ConnectedComponents::Connectivity::Four);
    std::cout << std::format("Labels (Four): {}\n", cm4.count());

    // ── DistanceTransform: L2 (default) ──────────────────────────────────────
    Image<Float32> dt = binary | DistanceTransform{};
    cv::Mat dt_norm;
    cv::normalize(dt.mat(), dt_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    show("4 — DistanceTransform L2 (normalised)", dt_norm);

    // ── DistanceTransform: L1 ────────────────────────────────────────────────
    Image<Float32> dt_l1 = binary | DistanceTransform{}.dist_type(DistanceTransform::DistType::L1);
    cv::Mat dt_l1_norm;
    cv::normalize(dt_l1.mat(), dt_l1_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    show("5 — DistanceTransform L1 (normalised)", dt_l1_norm);

    // ── Pipeline: threshold → ConnectedComponents + bounding boxes ───────────
    ComponentMap cm2 = bgr | ToGray{} | Threshold{}.value(100).mode(ThresholdMode::Binary)
                           | ConnectedComponents{};
    Image<BGR> annotated(raw.clone());
    for (int i = 1; i < cm2.count(); ++i) {
        cv::Scalar color(
            static_cast<double>((i * 90) % 256),
            static_cast<double>((i * 160) % 256),
            static_cast<double>((i * 220) % 256));
        cv::rectangle(annotated.mat(), cm2.bounding_rect(i), color, 2);
    }
    show("6 — ConnectedComponents bounding boxes", annotated.mat());

    std::cout << "Done.\n";
    return 0;
}
