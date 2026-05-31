// examples/core/demo_feature_ops.cpp
//
// Demo: HoughLinesP, HoughCircles, MatchTemplate

#include <format>
#include <iostream>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

int main() {
    // ── HoughLinesP ───────────────────────────────────────────────────────────
    cv::Mat edge_mat(200, 200, CV_8UC1, cv::Scalar(0));
    cv::line(edge_mat, {10, 10}, {190, 190}, cv::Scalar(255), 1);
    cv::line(edge_mat, {10, 190}, {190, 10}, cv::Scalar(255), 1);
    Image<Gray> edges(edge_mat);

    auto lines = HoughLinesP()
        .threshold(30)
        .min_line_length(50.0)(edges);
    std::cout << std::format("HoughLinesP detected {} line(s)\n", lines.size());
    for (const auto& l : lines)
        std::cout << std::format("  line: ({},{}) -> ({},{})\n",
                                 l[0], l[1], l[2], l[3]);

    // ── HoughCircles ──────────────────────────────────────────────────────────
    cv::Mat circle_mat(300, 300, CV_8UC1, cv::Scalar(255));
    cv::circle(circle_mat, {150, 150}, 60, cv::Scalar(0), 2);
    cv::GaussianBlur(circle_mat, circle_mat, {9, 9}, 2.0);
    Image<Gray> circle_img(circle_mat);

    auto circles = HoughCircles()
        .min_dist(100.0)
        .param1(50.0)
        .param2(20.0)
        .min_radius(40)
        .max_radius(80)(circle_img);
    std::cout << std::format("HoughCircles detected {} circle(s)\n", circles.size());
    for (const auto& c : circles)
        std::cout << std::format("  circle: centre=({:.0f},{:.0f}) r={:.0f}\n",
                                 c[0], c[1], c[2]);

    // ── MatchTemplate ─────────────────────────────────────────────────────────
    cv::Mat scene(200, 200, CV_8UC3, cv::Scalar(50, 50, 50));
    cv::rectangle(scene, {80, 80}, {120, 120}, cv::Scalar(200, 100, 50), -1);
    Image<BGR> scene_img(scene);

    cv::Mat templ_mat = scene(cv::Rect(80, 80, 40, 40));
    Image<BGR> templ(templ_mat.clone());

    auto [pt, score] = MatchTemplate{}(scene_img, templ);
    std::cout << std::format("MatchTemplate best match at ({},{}) score={:.4f}\n",
                             pt.x, pt.y, score);

    std::cout << "demo_feature_ops: OK\n";
    return 0;
}
