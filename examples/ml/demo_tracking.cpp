// examples/ml/demo_tracking.cpp
#include <iostream>
#include "improc/ml/tracking/tracking.hpp"

int main() {
    using namespace improc::ml;

    // Simulate a car moving right over 5 frames; one frame with only low-conf detection
    ByteTracker tracker;
    tracker.min_hits(1).max_age(3)
           .high_conf_threshold(0.6f).low_conf_threshold(0.1f);

    TrackingEval eval;

    struct Frame { float x; float conf; };
    const Frame frames[] = {
        {0.0f,  0.9f},
        {5.0f,  0.9f},
        {10.0f, 0.3f},  // low-conf: ByteTracker recovers via Stage 2
        {15.0f, 0.9f},
        {20.0f, 0.9f},
    };

    for (int f = 0; f < 5; ++f) {
        Detection det{cv::Rect2f(frames[f].x, 100, 50, 50), 0, frames[f].conf, "car"};
        auto tracks = tracker.update({det});

        TrackGT gt{0, BBox{cv::Rect2f(frames[f].x, 100, 50, 50), 0, "car"}};
        eval.update(tracks, {gt});

        for (const auto& t : tracks)
            std::cout << "frame " << f << ": track " << t.id
                      << " @ x=" << t.bbox.box.x
                      << " conf_bucket=" << (frames[f].conf >= 0.6f ? "high" : "low") << '\n';
    }

    auto m = eval.compute();
    std::cout << "\nMOTA: " << m.MOTA
              << "  MOTP: " << m.MOTP
              << "  IDF1: " << m.IDF1
              << "  IDSW: " << m.IDSW << '\n';
    return 0;
}
