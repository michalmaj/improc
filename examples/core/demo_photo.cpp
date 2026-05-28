// examples/core/demo_photo.cpp
//
// Demo: EdgePreservingFilter, DetailEnhance, Stylize, PencilSketch, SeamlessClone
//
// Generates a colourful synthetic image, applies each photo/creative op,
// shows results in named windows. Press any key to advance.

#include "improc/core/pipeline.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace improc::core;

static void show(const std::string& t, const cv::Mat& m) {
    cv::imshow(t, m); cv::waitKey(0);
}

int main() {
    // ── Synthetic colourful scene ─────────────────────────────────────────────
    cv::Mat raw(400, 500, CV_8UC3, cv::Scalar(100, 150, 200));
    cv::rectangle(raw, {30,  30},  {200, 180}, {200,  80,  30}, -1);
    cv::circle(raw,    {380, 200}, 90,          { 30, 200,  80}, -1);
    cv::rectangle(raw, {80,  250}, {300, 380},  { 50,  50, 220}, -1);
    cv::circle(raw,    {420,  80}, 50,           {220, 200,  50}, -1);

    // Add some texture
    cv::Mat noise(raw.size(), CV_8UC3);
    cv::randu(noise, 0, 20);
    raw += noise;

    Image<BGR> src(raw);
    show("0 — Source", raw);

    // ── EdgePreservingFilter ──────────────────────────────────────────────────
    std::cout << "\n--- EdgePreservingFilter ---\n";
    Image<BGR> ep = src | EdgePreservingFilter{}
        .sigma_s(60.f)                                     // spatial scale (default: 60)
        .sigma_r(0.4f)                                     // range scale (default: 0.4)
        .filter(EdgePreservingFilter::Filter::Recursive);   // Recursive (default) or NormConv
    show("1 — EdgePreservingFilter (Recursive)", ep.mat());

    // ── DetailEnhance ─────────────────────────────────────────────────────────
    std::cout << "\n--- DetailEnhance ---\n";
    Image<BGR> detail = src | DetailEnhance{}
        .sigma_s(10.f)    // spatial scale (default: 10)
        .sigma_r(0.15f);  // range scale (default: 0.15)
    show("2 — DetailEnhance", detail.mat());

    // ── Stylize ───────────────────────────────────────────────────────────────
    std::cout << "\n--- Stylize ---\n";
    Image<BGR> styled = src | Stylize{}
        .sigma_s(60.f)    // spatial scale (default: 60)
        .sigma_r(0.45f);  // range scale (default: 0.45)
    show("3 — Stylize (oil-painting effect)", styled.mat());

    // ── PencilSketch ──────────────────────────────────────────────────────────
    std::cout << "\n--- PencilSketch ---\n";
    PencilSketchResult sk = PencilSketch{}
        .sigma_s(60.f)        // spatial scale (default: 60)
        .sigma_r(0.07f)       // range scale (default: 0.07)
        .shade_factor(0.05f)  // pencil shading brightness (default: 0.05)
        (src);
    show("4a — PencilSketch (gray)", sk.gray.mat());
    show("4b — PencilSketch (colour)", sk.color.mat());

    // ── SeamlessClone ─────────────────────────────────────────────────────────
    std::cout << "\n--- SeamlessClone ---\n";
    // Paste a circle patch from source onto a grey background
    cv::Mat bg(400, 500, CV_8UC3, cv::Scalar(180, 180, 180));
    cv::Mat patch = raw(cv::Rect{290, 110, 100, 100}).clone();
    cv::Mat dst_mat = bg.clone();
    patch.copyTo(dst_mat(cv::Rect{200, 150, 100, 100}));
    Image<BGR> dst(dst_mat);

    cv::Mat mask(100, 100, CV_8U, cv::Scalar(255));
    Image<Gray> mask_img(mask);

    Image<BGR> cloned = SeamlessClone{}
        .mode(SeamlessClone::Mode::Normal)  // Normal, Mixed, or Monochrome (default: Normal)
        (Image<BGR>{patch}, dst, mask_img, {250, 200});
    show("5 — SeamlessClone (Poisson blending)", cloned.mat());

    std::cout << "Done.\n";
    return 0;
}
