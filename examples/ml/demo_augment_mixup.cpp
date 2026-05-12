#include <iostream>
#include <random>
#include "improc/ml/augmentation.hpp"

int main() {
    using namespace improc::core;
    using namespace improc::ml;

    std::mt19937 rng(42);

    cv::Mat mat_a(480, 640, CV_8UC3, cv::Scalar(50, 150, 250));
    cv::Mat mat_b(480, 640, CV_8UC3, cv::Scalar(250, 100, 50));

    // 3-class one-hot labels
    LabeledImage<BGR> a{Image<BGR>(mat_a), {1.0f, 0.0f, 0.0f}};  // class 0
    LabeledImage<BGR> b{Image<BGR>(mat_b), {0.0f, 1.0f, 0.0f}};  // class 1

    auto print_label = [](const char* name, const std::vector<float>& label) {
        std::cout << name << " label: [";
        for (std::size_t i = 0; i < label.size(); ++i)
            std::cout << label[i] << (i + 1 < label.size() ? ", " : "");
        std::cout << "]\n";
    };

    // --- MixUp ---
    auto mu = MixUp{}.alpha(0.4f)(a, b, rng);
    print_label("MixUp", mu.label);

    // --- CutMix ---
    auto cm = CutMix{}.alpha(1.0f)(a, b, rng);
    print_label("CutMix", cm.label);

    // --- MixCompose: sequential MixUp → CutMix ---
    MixCompose<BGR> pipe;
    pipe
        .add([](auto a, const auto& b, auto& r){
            return MixUp{}.alpha(0.4f)(std::move(a), b, r);
        })
        .add([](auto a, const auto& b, auto& r){
            return CutMix{}.alpha(1.0f)(std::move(a), b, r);
        });

    auto result  = pipe(a, b, rng);
    auto result2 = a | pipe.bind(b, rng);

    print_label("Compose (direct)", result.label);
    print_label("Compose (pipeline)", result2.label);

    return 0;
}
