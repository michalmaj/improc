// examples/ml/demo_detection_eval.cpp
#include <iostream>
#include "improc/ml/eval/detection.hpp"

int main() {
    using namespace improc::ml;

    DetectionEval eval;

    // Image 1: perfect match
    eval.update(
        {{cv::Rect2f(10, 10, 50, 50), 0, 0.95f, "cat"}},
        {{cv::Rect2f(10, 10, 50, 50), 0, "cat"}}
    );
    // Image 2: slight offset (still high IoU)
    eval.update(
        {{cv::Rect2f(12, 12, 50, 50), 0, 0.88f, "cat"}},
        {{cv::Rect2f(10, 10, 50, 50), 0, "cat"}}
    );
    // Image 3: missed
    eval.update({}, {{cv::Rect2f(10, 10, 50, 50), 0, "cat"}});

    auto m = eval.compute();
    std::cout << "mAP@0.50:     " << m.mAP_50    << '\n'
              << "mAP@0.50:95:  " << m.mAP_50_95 << '\n'
              << "AP[cat]:      " << m.per_class_AP.at("cat") << '\n';
}
