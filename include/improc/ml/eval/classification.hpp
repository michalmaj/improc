// include/improc/ml/eval/classification.hpp
#pragma once
#include <map>
#include <span>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace improc::ml {

/// @brief Computes overall accuracy: correct / total.
/// @return Value in [0, 1]. Returns 0 if predictions is empty.
[[nodiscard]] float accuracy(std::span<const int> preds, std::span<const int> gts);

/// @brief Computes precision for a specific class.
/// @return Value in [0, 1].
[[nodiscard]] float precision_score(std::span<const int> preds, std::span<const int> gts, int class_id);

/// @brief Computes recall for a specific class.
/// @return Value in [0, 1].
[[nodiscard]] float recall_score(std::span<const int> preds, std::span<const int> gts, int class_id);

/// @brief Computes F1 score for a specific class.
/// @return Value in [0, 1].
[[nodiscard]] float f1_score(std::span<const int> preds, std::span<const int> gts, int class_id);

/**
 * @brief Dense confusion matrix as a `num_classes × num_classes` OpenCV matrix.
 *
 * `mat(i, j)` is the count of samples with true class `i` predicted as class `j`.
 */
struct ConfusionMatrix {
    cv::Mat_<int> mat;   ///< CV_32S confusion matrix (num_classes × num_classes).
    int num_classes = 0; ///< Number of classes.
};

/**
 * @brief Per-class and aggregate classification metrics computed from a confusion matrix.
 */
struct ClassMetrics {
    float accuracy = 0.0f;                            ///< Overall accuracy in [0, 1].
    std::map<std::string, float> per_class_precision; ///< Precision per class name.
    std::map<std::string, float> per_class_recall;    ///< Recall per class name.
    std::map<std::string, float> per_class_f1;        ///< F1 score per class name.
    ConfusionMatrix confusion_matrix;                 ///< Full confusion matrix.
};

/**
 * @brief Stateful accumulator that aggregates predictions and computes `ClassMetrics`.
 *
 * @code
 * ClassEval eval;
 * eval.class_names({"cat", "dog", "bird"});
 * eval.update(pred_class, gt_class);
 * auto metrics = eval.compute();
 * @endcode
 */
struct ClassEval {
    /// @brief Sets the class names. Must be called before `update()`.
    ClassEval& class_names(std::vector<std::string> names);

    /// @brief Appends a single prediction and ground-truth label.
    void update(int pred_class, int gt_class);
    /// @brief Computes and returns the classification metrics from all accumulated data.
    /// @return `ClassMetrics` aggregating all calls to `update()`.
    [[nodiscard]] ClassMetrics compute() const;
    /// @brief Resets all prediction data. Class name mappings are preserved.
    void reset();

private:
    cv::Mat_<int>            mat_;
    std::vector<std::string> names_;
};

} // namespace improc::ml
