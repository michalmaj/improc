// examples/core/demo_pyramid_drawing.cpp
//
// Demo: PyrDown, PyrUp, DrawText, DrawLine, DrawCircle, DrawRectangle
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
    // ── Source image ──────────────────────────────────────────────────────────
    cv::Mat raw(300, 400, CV_8UC3, cv::Scalar(40, 40, 80));
    cv::rectangle(raw, {50, 50},  {180, 150}, {200, 200, 200}, -1);
    cv::circle(raw,   {300, 150}, 70,          {180, 180, 180}, -1);
    cv::line(raw,     {0, 250},   {400, 250},  {220, 220, 220},  3);
    Image<BGR> bgr(raw);

    show("0 — Source (400×300)", bgr.mat());
    std::cout << std::format("Source: {}×{}\n", bgr.cols(), bgr.rows());

    // ── PyrDown / PyrUp ───────────────────────────────────────────────────────
    std::cout << "\n--- PyrDown / PyrUp ---\n";

    Image<BGR> down1 = bgr | PyrDown{};
    std::cout << std::format("PyrDown×1: {}×{} (expected 200×150)\n",
        down1.cols(), down1.rows());
    show("1a — PyrDown×1 (200×150)", down1.mat());

    Image<BGR> down2 = bgr | PyrDown{} | PyrDown{};
    std::cout << std::format("PyrDown×2: {}×{} (expected 100×75)\n",
        down2.cols(), down2.rows());
    show("1b — PyrDown×2 (100×75)", down2.mat());

    Image<BGR> up = down1 | PyrUp{};
    std::cout << std::format("PyrDown then PyrUp: {}×{} (expected 400×300)\n",
        up.cols(), up.rows());
    show("1c — PyrDown then PyrUp (400×300, blurred)", up.mat());

    Image<Gray> gray = bgr | ToGray{};
    Image<Gray> down_gray = gray | PyrDown{};
    std::cout << std::format("PyrDown on Gray: {}×{}\n",
        down_gray.cols(), down_gray.rows());
    show("1d — PyrDown on Gray (200×150)", down_gray.mat());

    // ── DrawText ──────────────────────────────────────────────────────────────
    std::cout << "\n--- DrawText ---\n";

    Image<BGR> t1 = bgr | DrawText{"Hello, improc!"};
    show("2a — DrawText default (top-left, green)", t1.mat());

    Image<BGR> t2 = bgr | DrawText{"Score: 0.95"}
        .position({10, 270})
        .font_scale(0.8)
        .color({0, 255, 255})
        .thickness(2);
    show("2b — DrawText custom (bottom, yellow, thick)", t2.mat());

    // ── DrawLine ──────────────────────────────────────────────────────────────
    std::cout << "\n--- DrawLine ---\n";

    Image<BGR> l1 = bgr | DrawLine{{0, 0}, {400, 300}};
    show("3a — DrawLine diagonal (green, default)", l1.mat());

    Image<BGR> l2 = bgr | DrawLine{{0, 150}, {400, 150}}.color({0, 0, 255}).thickness(3);
    show("3b — DrawLine horizontal (red, thick)", l2.mat());

    // ── DrawCircle ────────────────────────────────────────────────────────────
    std::cout << "\n--- DrawCircle ---\n";

    Image<BGR> c1 = bgr | DrawCircle{{200, 150}, 80};
    show("4a — DrawCircle outline (green, r=80)", c1.mat());

    Image<BGR> c2 = bgr | DrawCircle{{200, 150}, 50}.color({0, 0, 255}).thickness(-1);
    show("4b — DrawCircle filled (red, r=50)", c2.mat());

    // ── DrawRectangle ─────────────────────────────────────────────────────────
    std::cout << "\n--- DrawRectangle ---\n";

    Image<BGR> r1 = bgr | DrawRectangle{cv::Rect{40, 40, 200, 140}};
    show("5a — DrawRectangle outline (green)", r1.mat());

    Image<BGR> r2 = bgr | DrawRectangle{cv::Rect{100, 100, 80, 60}}
        .color({255, 0, 0}).thickness(-1);
    show("5b — DrawRectangle filled (blue)", r2.mat());

    // ── Pipeline chaining ─────────────────────────────────────────────────────
    std::cout << "\n--- Chained pipeline ---\n";

    Image<BGR> annotated = bgr
        | DrawRectangle{cv::Rect{50, 50, 130, 100}}.color({0, 255, 255})
        | DrawCircle{{300, 150}, 70}.color({0, 165, 255})
        | DrawText{"detected"}.position({55, 45}).font_scale(0.5).color({0, 255, 255})
        | DrawLine{{0, 250}, {400, 250}}.color({255, 255, 0}).thickness(2);
    show("6 — All ops chained", annotated.mat());

    std::cout << "Done.\n";
    return 0;
}
