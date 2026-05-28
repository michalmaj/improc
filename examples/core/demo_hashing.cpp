// examples/core/demo_hashing.cpp
//
// Demo: AverageHash, PHash, MarrHildrethHash, RadialVarianceHash,
//       ColorMomentHash, BlockMeanHash and their ::distance() methods
//
// Generates three images: original, slightly modified (brightness +10),
// and completely different. Prints a hash distance matrix for all pairs
// and all six algorithms.

#include "improc/core/pipeline.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <iomanip>
#include <string>

using namespace improc::core;

int main() {
    // ── Three test images ─────────────────────────────────────────────────────
    cv::Mat base(200, 200, CV_8UC3, cv::Scalar(100, 130, 160));
    cv::rectangle(base, {20, 20}, {90, 90},   {220,  80,  60}, -1);
    cv::circle(base,    {150, 100}, 50,         {60,  200, 100}, -1);

    // Image A: original
    Image<BGR> img_a(base);

    // Image B: slightly modified (brightness +10) — should score as "near-duplicate"
    cv::Mat modified = base + cv::Scalar(10, 10, 10);
    Image<BGR> img_b(modified);

    // Image C: completely different
    cv::Mat diff(200, 200, CV_8UC3, cv::Scalar(200, 50, 50));
    cv::circle(diff, {100, 100}, 80, {50, 50, 200}, -1);
    Image<BGR> img_c(diff);

    // ── Compute all hashes ────────────────────────────────────────────────────
    struct Algo {
        std::string name;
        cv::Mat ha, hb, hc;
        double dist_ab, dist_ac, dist_bc;
    };

    std::vector<Algo> algos;

    { Algo a{"AverageHash"};
      a.ha = AverageHash{}(img_a); a.hb = AverageHash{}(img_b); a.hc = AverageHash{}(img_c);
      a.dist_ab = AverageHash::distance(a.ha, a.hb);
      a.dist_ac = AverageHash::distance(a.ha, a.hc);
      a.dist_bc = AverageHash::distance(a.hb, a.hc);
      algos.push_back(a); }

    { Algo a{"PHash"};
      a.ha = PHash{}(img_a); a.hb = PHash{}(img_b); a.hc = PHash{}(img_c);
      a.dist_ab = PHash::distance(a.ha, a.hb);
      a.dist_ac = PHash::distance(a.ha, a.hc);
      a.dist_bc = PHash::distance(a.hb, a.hc);
      algos.push_back(a); }

    { Algo a{"MarrHildrethHash"};
      a.ha = MarrHildrethHash{}(img_a); a.hb = MarrHildrethHash{}(img_b); a.hc = MarrHildrethHash{}(img_c);
      a.dist_ab = MarrHildrethHash::distance(a.ha, a.hb);
      a.dist_ac = MarrHildrethHash::distance(a.ha, a.hc);
      a.dist_bc = MarrHildrethHash::distance(a.hb, a.hc);
      algos.push_back(a); }

    { Algo a{"RadialVarianceHash"};
      a.ha = RadialVarianceHash{}(img_a); a.hb = RadialVarianceHash{}(img_b); a.hc = RadialVarianceHash{}(img_c);
      a.dist_ab = RadialVarianceHash::distance(a.ha, a.hb);
      a.dist_ac = RadialVarianceHash::distance(a.ha, a.hc);
      a.dist_bc = RadialVarianceHash::distance(a.hb, a.hc);
      algos.push_back(a); }

    { Algo a{"ColorMomentHash"};
      a.ha = ColorMomentHash{}(img_a); a.hb = ColorMomentHash{}(img_b); a.hc = ColorMomentHash{}(img_c);
      a.dist_ab = ColorMomentHash::distance(a.ha, a.hb);
      a.dist_ac = ColorMomentHash::distance(a.ha, a.hc);
      a.dist_bc = ColorMomentHash::distance(a.hb, a.hc);
      algos.push_back(a); }

    { Algo a{"BlockMeanHash"};
      a.ha = BlockMeanHash{}(img_a); a.hb = BlockMeanHash{}(img_b); a.hc = BlockMeanHash{}(img_c);
      a.dist_ab = BlockMeanHash::distance(a.ha, a.hb);
      a.dist_ac = BlockMeanHash::distance(a.ha, a.hc);
      a.dist_bc = BlockMeanHash::distance(a.hb, a.hc);
      algos.push_back(a); }

    // ── Print distance matrix ─────────────────────────────────────────────────
    std::cout << "\n"
              << std::left << std::setw(20) << "Algorithm"
              << std::right
              << std::setw(12) << "A↔B(similar)"
              << std::setw(12) << "A↔C(differ)"
              << std::setw(12) << "B↔C(differ)" << "\n";
    std::cout << std::string(56, '-') << "\n";
    for (const auto& a : algos) {
        std::cout << std::left << std::setw(20) << a.name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << a.dist_ab
                  << std::setw(12) << a.dist_ac
                  << std::setw(12) << a.dist_bc << "\n";
    }
    std::cout << "\n(Lower A↔B + higher A↔C = better discrimination)\n";
    std::cout << "Done.\n";
    return 0;
}
