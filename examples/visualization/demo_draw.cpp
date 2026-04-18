// examples/visualization/demo_draw.cpp
//
// Demo: DrawBoundingBoxes — annotating detection results on images.
//
// Uses synthetic detections so no model or camera is required.
// Press any key to advance between windows.

#include "improc/core/pipeline.hpp"
#include "improc/visualization/draw.hpp"
#include "improc/ml/result_types.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace improc::core;
using namespace improc::visualization;
using improc::ml::Detection;

static void show(const std::string& title, const cv::Mat& mat) {
    cv::imshow(title, mat);
    cv::waitKey(0);
}

static Detection make_det(float x, float y, float w, float h,
                           int class_id, std::string label, float conf) {
    Detection d;
    d.box        = {x, y, w, h};
    d.class_id   = class_id;
    d.label      = std::move(label);
    d.confidence = conf;
    return d;
}

int main() {
    // ── Background: simple scene with coloured regions ─────────────────────
    cv::Mat raw(480, 640, CV_8UC3, cv::Scalar(40, 40, 40));
    // "objects" — filled rectangles in different colours
    cv::rectangle(raw, {30,  40},  {180, 180}, {60,  130, 200}, -1);   // blue-ish
    cv::rectangle(raw, {220, 80},  {400, 260}, {50,  180,  70}, -1);   // green-ish
    cv::circle(raw,    {530, 150},  80,         {180,  60,  60}, -1);   // red-ish
    cv::rectangle(raw, {60,  280}, {260, 420},  {170, 130,  30}, -1);  // yellow-ish
    cv::ellipse(raw,   {450, 360},  {100, 60}, 0, 0, 360, {120, 60, 170}, -1); // purple-ish

    Image<BGR> scene(raw);
    show("0 — Scene (no annotations)", scene.mat());

    // ── 1. Basic annotations ──────────────────────────────────────────────────
    std::cout << "\n--- Basic annotations ---\n";

    std::vector<Detection> dets = {
        make_det( 30,  40, 150, 140, 0, "person",  0.92f),
        make_det(220,  80, 180, 180, 1, "car",     0.87f),
        make_det(450,  70, 160, 160, 2, "bicycle", 0.75f),
        make_det( 60, 280, 200, 140, 3, "dog",     0.83f),
        make_det(350, 300, 200, 120, 4, "cat",     0.61f),
    };

    Image<BGR> annotated = scene | DrawBoundingBoxes{dets};
    show("1  — Default (green, thickness=2, label+confidence)", annotated.mat());

    // ── 2. Display variants ───────────────────────────────────────────────────
    std::cout << "\n--- Display variants ---\n";

    Image<BGR> label_only = scene | DrawBoundingBoxes{dets}.show_confidence(false);
    show("2a — Label only (no confidence)", label_only.mat());

    Image<BGR> conf_only = scene | DrawBoundingBoxes{dets}.show_label(false);
    show("2b — Confidence only (no label)", conf_only.mat());

    Image<BGR> boxes_only = scene | DrawBoundingBoxes{dets}
        .show_label(false).show_confidence(false);
    show("2c — Boxes only", boxes_only.mat());

    // ── 3. Custom appearance ──────────────────────────────────────────────────
    std::cout << "\n--- Custom appearance ---\n";

    Image<BGR> red_thick = scene | DrawBoundingBoxes{dets}
        .color({0, 0, 255})
        .thickness(3)
        .font_scale(0.6);
    show("3a — Red boxes, thickness=3, font_scale=0.6", red_thick.mat());

    Image<BGR> cyan_thin = scene | DrawBoundingBoxes{dets}
        .color({255, 255, 0})
        .thickness(1)
        .font_scale(0.35)
        .show_confidence(false);
    show("3b — Cyan, thin, labels only", cyan_thin.mat());

    // ── 4. Source image unchanged ─────────────────────────────────────────────
    std::cout << "\n--- Source unchanged ---\n";

    Image<BGR> before = scene;
    DrawBoundingBoxes{dets}(scene);  // result discarded — scene must be untouched
    cv::Mat diff;
    cv::absdiff(before.mat(), scene.mat(), diff);
    std::cout << "  Diff sum after DrawBoundingBoxes: "
              << cv::sum(diff)[0] << " (should be 0)\n";

    // ── 5. Low-confidence filter ──────────────────────────────────────────────
    std::cout << "\n--- Filter by confidence threshold ---\n";

    std::vector<Detection> high_conf;
    for (const auto& d : dets)
        if (d.confidence >= 0.80f)
            high_conf.push_back(d);

    Image<BGR> filtered = scene | DrawBoundingBoxes{high_conf};
    std::cout << "  Showing " << high_conf.size() << "/" << dets.size()
              << " detections (conf >= 0.80)\n";
    show("5  — Only high-confidence detections (≥ 0.80)", filtered.mat());

    // ── 6. Pipeline: inference prep → annotate → display ──────────────────────
    std::cout << "\n--- Full pipeline ---\n";

    // Simulate: resize for inference, annotate, resize back for display
    std::vector<Detection> scaled_dets = {
        make_det( 7,  9,  30,  27, 0, "person",  0.91f),
        make_det(44,  16, 36,  36, 1, "car",     0.88f),
        make_det(12,  56, 40,  28, 3, "dog",     0.82f),
    };

    Image<BGR> pipeline_result =
        scene
        | Resize{}.width(128).height(96)
        | DrawBoundingBoxes{scaled_dets}.thickness(1).font_scale(0.3)
        | Resize{}.width(640).height(480);

    show("6  — Resize → Annotate → Resize back", pipeline_result.mat());

    std::cout << "Done.\n";
    return 0;
}
