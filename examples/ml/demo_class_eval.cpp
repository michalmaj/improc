// examples/ml/demo_class_eval.cpp
#include <iostream>
#include "improc/ml/eval/classification.hpp"

int main() {
    using namespace improc::ml;

    ClassEval eval;
    eval.class_names({"cat", "dog", "bird"});

    // Simulated predictions: mostly correct, some confusion
    eval.update(0, 0); eval.update(0, 0); eval.update(0, 0);  // cat correct x3
    eval.update(1, 1); eval.update(1, 1); eval.update(0, 1);  // dog 2 correct, 1 as cat
    eval.update(2, 2); eval.update(2, 2); eval.update(1, 2);  // bird 2 correct, 1 as dog

    auto m = eval.compute();
    std::cout << "Accuracy: " << m.accuracy << "\n\n";
    for (const auto& [cls, f1] : m.per_class_f1)
        std::cout << "F1[" << cls << "]: " << f1 << '\n';

    std::cout << "\nConfusion matrix:\n";
    for (int r = 0; r < m.confusion_matrix.mat.rows; ++r) {
        for (int c = 0; c < m.confusion_matrix.mat.cols; ++c)
            std::cout << m.confusion_matrix.mat(r, c) << ' ';
        std::cout << '\n';
    }
}
