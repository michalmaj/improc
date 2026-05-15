// examples/ml/demo_seg_eval.cpp
#include <iostream>
#include <opencv2/core.hpp>
#include "improc/ml/eval/segmentation.hpp"

int main() {
    using namespace improc::core;
    using namespace improc::ml;

    SegEval eval;
    eval.num_classes(3);

    // Synthesise 3 images: two perfect, one with noise
    cv::Mat perfect(64, 64, CV_8U, cv::Scalar(1));
    cv::Mat noisy = perfect.clone();
    noisy.rowRange(0, 32).setTo(2);  // top half predicted as class 2 instead of 1

    eval.update(Image<Gray>(perfect), Image<Gray>(perfect));
    eval.update(Image<Gray>(perfect), Image<Gray>(perfect));
    eval.update(Image<Gray>(noisy),   Image<Gray>(perfect));

    auto m = eval.compute();
    std::cout << "mIoU:        " << m.mIoU      << '\n'
              << "mean Dice:   " << m.mean_dice  << '\n';
    for (const auto& [cls, iou_val] : m.per_class_iou)
        std::cout << "  class " << cls << " IoU: " << iou_val << '\n';
}
