// examples/core/demo_hdr.cpp
//
// Demo: MergeHDR (Mertens exposure fusion), ToneMap (Reinhard), Stitch
//
// Synthesises a bracketed exposure set from a single image using gamma shifts,
// merges to a Float32C3 HDR image, tone-maps back to 8-bit BGR, then stitches
// two overlapping crops of the original to demonstrate Stitch.
// Press any key to advance.

#include "improc/core/pipeline.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace improc::core;

static void show(const std::string& t, const cv::Mat& m) {
    cv::imshow(t, m); cv::waitKey(0);
}

// Apply a gamma shift to simulate different exposures
static cv::Mat gamma_shift(const cv::Mat& src, double gamma) {
    cv::Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i)
        lut.at<uchar>(i) = cv::saturate_cast<uchar>(
            255.0 * std::pow(i / 255.0, gamma));
    cv::Mat dst;
    cv::LUT(src, lut, dst);
    return dst;
}

int main() {
    // ── Synthetic HDR scene: three exposure-bracketed frames ──────────────────
    cv::Mat raw(300, 400, CV_8UC3, cv::Scalar(80, 100, 120));
    cv::rectangle(raw, {20,  20},  {180, 150}, {220, 180,  60}, -1);
    cv::circle(raw,    {300, 150}, 80,          { 60, 180, 220}, -1);
    cv::rectangle(raw, {100, 180}, {380, 280},  {180,  60, 180}, -1);
    cv::Mat noise(raw.size(), CV_8UC3);
    cv::randu(noise, 0, 15);
    raw += noise;

    // Simulate under-, normal, and over-exposed frames
    std::vector<Image<BGR>> frames = {
        Image<BGR>{gamma_shift(raw, 0.5)},   // underexposed
        Image<BGR>{raw},                      // normal
        Image<BGR>{gamma_shift(raw, 2.0)}    // overexposed
    };
    show("1a — Underexposed frame",  frames[0].mat());
    show("1b — Normal frame",        frames[1].mat());
    show("1c — Overexposed frame",   frames[2].mat());

    // ── MergeHDR (Mertens) — no exposure times needed ─────────────────────────
    std::cout << "\n--- MergeHDR (Mertens) ---\n";
    Image<Float32C3> hdr = MergeHDR{}
        .method(MergeHDR::Method::Mertens)  // Mertens = exposure fusion (default)
        (frames);
    std::cout << "HDR type: CV_32FC3 = " << (hdr.mat().type() == CV_32FC3) << "\n";

    // ── ToneMap — Float32C3 → BGR ─────────────────────────────────────────────
    std::cout << "\n--- ToneMap (Reinhard) ---\n";
    Image<BGR> tm_reinhard = ToneMap{}
        .gamma(1.f)                               // gamma correction (default: 1)
        .algorithm(ToneMap::Algorithm::Reinhard)  // Reinhard, Linear, Drago, Mantiuk
        (hdr);
    show("2 — Tone-mapped (Reinhard)", tm_reinhard.mat());

    Image<BGR> tm_drago = ToneMap{}
        .gamma(1.f)                               // gamma correction (default: 1)
        .algorithm(ToneMap::Algorithm::Drago)     // Reinhard, Linear, Drago, Mantiuk
        (hdr);
    show("3 — Tone-mapped (Drago)", tm_drago.mat());

    // ── Stitch — panorama from two overlapping crops ───────────────────────────
    std::cout << "\n--- Stitch ---\n";
    Image<BGR> left_crop{raw(cv::Rect{0,   0, 250, 300}).clone()};
    Image<BGR> right_crop{raw(cv::Rect{150, 0, 250, 300}).clone()};

    StitchResult sr = Stitch{}
        .mode(Stitch::Mode::Panorama)   // Panorama (default) or Scans
        ({left_crop, right_crop});

    if (sr.ok) {
        std::cout << "Panorama size: " << sr.panorama.mat().cols
                  << "×" << sr.panorama.mat().rows << "\n";
        show("4 — Stitched panorama", sr.panorama.mat());
    } else {
        std::cout << "Stitch failed (synthetic crops may lack enough features)\n";
    }

    std::cout << "Done.\n";
    return 0;
}
