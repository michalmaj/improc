// include/improc/ml/eval/classification.hpp
#pragma once
#include <map>
#include <span>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace improc::ml {

[[nodiscard]] float accuracy(std::span<const int> preds, std::span<const int> gts);
[[nodiscard]] float precision_score(std::span<const int> preds, std::span<const int> gts, int class_id);
[[nodiscard]] float recall_score(std::span<const int> preds, std::span<const int> gts, int class_id);
[[nodiscard]] float f1_score(std::span<const int> preds, std::span<const int> gts, int class_id);

struct ConfusionMatrix {
    cv::Mat_<int> mat;
    int num_classes = 0;
};

struct ClassMetrics {
    float accuracy = 0.0f;
    std::map<std::string, float> per_class_precision;
    std::map<std::string, float> per_class_recall;
    std::map<std::string, float> per_class_f1;
    ConfusionMatrix confusion_matrix;
};

struct ClassEval {
    ClassEval& class_names(std::vector<std::string> names);

    void update(int pred_class, int gt_class);
    [[nodiscard]] ClassMetrics compute() const;
    void reset();

private:
    cv::Mat_<int>            mat_;
    std::vector<std::string> names_;
};

} // namespace improc::ml
