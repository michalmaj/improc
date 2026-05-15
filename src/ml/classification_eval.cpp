// src/ml/classification_eval.cpp
#include "improc/ml/eval/classification.hpp"
#include <algorithm>
#include <stdexcept>

namespace improc::ml {

namespace {

int count_tp(std::span<const int> preds, std::span<const int> gts, int cls) {
    int tp = 0;
    for (std::size_t i = 0; i < preds.size(); ++i)
        if (preds[i] == cls && gts[i] == cls) ++tp;
    return tp;
}
int col_sum(std::span<const int> preds, int cls) {
    int s = 0; for (int p : preds) if (p == cls) ++s; return s;
}
int row_sum(std::span<const int> gts, int cls) {
    int s = 0; for (int g : gts) if (g == cls) ++s; return s;
}

} // namespace

float accuracy(std::span<const int> preds, std::span<const int> gts) {
    if (preds.empty()) return 0.0f;
    int correct = 0;
    for (std::size_t i = 0; i < preds.size(); ++i)
        if (preds[i] == gts[i]) ++correct;
    return static_cast<float>(correct) / static_cast<float>(preds.size());
}

float precision_score(std::span<const int> preds, std::span<const int> gts, int class_id) {
    int denom = col_sum(preds, class_id);
    return denom == 0 ? 0.0f
        : static_cast<float>(count_tp(preds, gts, class_id)) / denom;
}

float recall_score(std::span<const int> preds, std::span<const int> gts, int class_id) {
    int denom = row_sum(gts, class_id);
    return denom == 0 ? 0.0f
        : static_cast<float>(count_tp(preds, gts, class_id)) / denom;
}

float f1_score(std::span<const int> preds, std::span<const int> gts, int class_id) {
    float prec = precision_score(preds, gts, class_id);
    float rec  = recall_score(preds, gts, class_id);
    return (prec + rec) == 0.0f ? 0.0f : 2.0f * prec * rec / (prec + rec);
}

ClassEval& ClassEval::class_names(std::vector<std::string> names) {
    names_ = std::move(names);
    int n = static_cast<int>(names_.size());
    mat_ = cv::Mat_<int>(n, n, 0);
    return *this;
}

void ClassEval::update(int pred_class, int gt_class) {
    int sz = std::max({mat_.rows, pred_class + 1, gt_class + 1});
    if (sz > mat_.rows) {
        cv::Mat_<int> bigger(sz, sz, 0);
        if (!mat_.empty()) mat_.copyTo(bigger(cv::Rect(0, 0, mat_.cols, mat_.rows)));
        mat_ = bigger;
    }
    mat_(gt_class, pred_class)++;
}

ClassMetrics ClassEval::compute() const {
    if (mat_.empty()) return {};
    int n = mat_.rows;
    ClassMetrics m;
    m.confusion_matrix = {mat_.clone(), n};

    auto key = [&](int i) -> std::string {
        return i < static_cast<int>(names_.size()) ? names_[i] : std::to_string(i);
    };

    int total = 0, correct = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) total += mat_(i, j);
        correct += mat_(i, i);
    }
    m.accuracy = total > 0 ? static_cast<float>(correct) / total : 0.0f;

    for (int c = 0; c < n; ++c) {
        int tp = mat_(c, c);
        int col = 0, row = 0;
        for (int i = 0; i < n; ++i) col += mat_(i, c);
        for (int j = 0; j < n; ++j) row += mat_(c, j);

        float prec = col > 0 ? static_cast<float>(tp) / col : 0.0f;
        float rec  = row > 0 ? static_cast<float>(tp) / row : 0.0f;
        float f1   = (prec + rec) > 0.0f ? 2.0f * prec * rec / (prec + rec) : 0.0f;

        std::string k = key(c);
        m.per_class_precision[k] = prec;
        m.per_class_recall[k]    = rec;
        m.per_class_f1[k]        = f1;
    }
    return m;
}
void ClassEval::reset() { if (!mat_.empty()) mat_.setTo(0); }

} // namespace improc::ml
