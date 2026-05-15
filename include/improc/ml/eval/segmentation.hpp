// include/improc/ml/eval/segmentation.hpp
#pragma once
#include <map>
#include <stdexcept>
#include <vector>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"

namespace improc::ml {

using improc::core::Gray;
using improc::core::Image;

// Pixel IoU for class_id over two single-channel masks.
// Pixels where gt == 255 (void label) are ignored.
[[nodiscard]] float pixel_iou(const Image<Gray>& pred, const Image<Gray>& gt, int class_id);

// Dice coefficient for class_id. Same void-pixel rule as pixel_iou.
[[nodiscard]] float dice(const Image<Gray>& pred, const Image<Gray>& gt, int class_id);

struct SegMetrics {
    float mIoU      = 0.0f;
    float mean_dice = 0.0f;
    std::map<int, float> per_class_iou;
    std::map<int, float> per_class_dice;
};

struct SegEval {
    SegEval& num_classes(int n) {
        num_classes_ = n;
        tp_.assign(n, 0);
        fp_.assign(n, 0);
        fn_.assign(n, 0);
        return *this;
    }

    void update(const Image<Gray>& pred_mask, const Image<Gray>& gt_mask);
    [[nodiscard]] SegMetrics compute() const;
    void reset();

private:
    int num_classes_ = 0;
    std::vector<int64_t> tp_, fp_, fn_;
};

} // namespace improc::ml
