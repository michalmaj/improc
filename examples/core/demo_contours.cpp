// examples/core/demo_contours.cpp
//
// Demo: ContourSet, FindContours, DrawContours
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
    // ── Source: shapes on a dark background ──────────────────────────────────
    cv::Mat raw(300, 400, CV_8UC3, cv::Scalar(30, 30, 30));
    cv::rectangle(raw, {40,  40},  {160, 140}, cv::Scalar(220, 220, 220), -1);
    cv::circle(raw,   {280, 100}, 60,           cv::Scalar(200, 200, 200), -1);
    cv::ellipse(raw,  {200, 230}, {80, 40}, 0, 0, 360, cv::Scalar(180, 180, 180), -1);
    Image<BGR>  bgr(raw);
    Image<Gray> gray = bgr | ToGray{};

    show("0a — Source BGR",  bgr.mat());
    show("0b — Source Gray", gray.mat());

    // ── Threshold to get binary image ─────────────────────────────────────────
    Image<Gray> binary = gray | Threshold{}.value(100).mode(ThresholdMode::Binary);
    show("1 — Binary (threshold=100)", binary.mat());

    // ── FindContours: External (default) ─────────────────────────────────────
    ContourSet cs = binary | FindContours{};
    std::cout << std::format("External contours: {}\n", cs.size());
    for (std::size_t i = 0; i < cs.size(); ++i)
        std::cout << std::format("  [{}] area={:.0f}  perimeter={:.0f}  bbox={}x{}\n",
            i, cs.area(i), cs.perimeter(i),
            cs.bounding_rect(i).width, cs.bounding_rect(i).height);

    Image<BGR> drawn_all = bgr | DrawContours{cs}.color({0, 255, 0}).thickness(2);
    show("2 — DrawContours all (green, thickness=2)", drawn_all.mat());

    // ── Draw individual contours ──────────────────────────────────────────────
    if (cs.size() >= 1) {
        Image<BGR> one = bgr | DrawContours{cs}.index(0).color({0, 0, 255}).thickness(2);
        show("3 — DrawContours index=0 (red)", one.mat());
    }

    // ── Fill contours ─────────────────────────────────────────────────────────
    Image<BGR> filled = bgr | DrawContours{cs}.thickness(-1).color({0, 128, 255});
    show("4 — DrawContours filled (orange)", filled.mat());

    // ── FindContours: Tree mode ───────────────────────────────────────────────
    ContourSet tree_cs = binary | FindContours{}.mode(FindContours::Mode::Tree);
    std::cout << std::format("Tree contours: {}\n", tree_cs.size());
    Image<BGR> tree_drawn = bgr | DrawContours{tree_cs}.color({255, 255, 0}).thickness(1);
    show("5 — FindContours Tree mode (yellow)", tree_drawn.mat());

    // ── FindContours: Method::None (all boundary points) ─────────────────────
    ContourSet none_cs = binary | FindContours{}.method(FindContours::Method::None);
    if (none_cs.size() > 0 && cs.size() > 0)
        std::cout << std::format("Method::None pts[0]={} vs Simple pts[0]={}\n",
            none_cs.contours[0].size(), cs.contours[0].size());

    // ── Pipeline chaining ─────────────────────────────────────────────────────
    Image<BGR> result = bgr
        | ToGray{}
        | Threshold{}.value(100).mode(ThresholdMode::Binary)
        | [](Image<Gray> g) {
            ContourSet c = g | FindContours{};
            Image<BGR> base(cv::Mat(g.rows(), g.cols(), CV_8UC3, cv::Scalar(30, 30, 30)));
            return base | DrawContours{c}.color({0, 255, 128}).thickness(2);
          };
    show("6 — Chained: ToGray | Threshold | FindContours | DrawContours", result.mat());

    std::cout << "Done.\n";
    return 0;
}
