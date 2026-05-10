#include <iostream>
#include <random>
#include "improc/ml/augmentation.hpp"

int main() {
    using namespace improc::core;
    using namespace improc::ml;

    std::mt19937 rng(42);

    cv::Mat mat(480, 640, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(mat);

    BBox cat{cv::Rect2f(50.f,  50.f,  200.f, 150.f), 0, "cat"};
    BBox dog{cv::Rect2f(300.f, 100.f, 180.f, 200.f), 1, "dog"};
    AnnotatedImage<BGR> ann{img, {cat, dog}};

    std::cout << "Original: " << ann.boxes.size() << " boxes\n";
    for (const auto& b : ann.boxes)
        std::cout << "  [" << b.label << " cls=" << b.class_id << "] " << b.box << "\n";

    BBoxCompose<BGR> pipeline;
    pipeline
        .add([](auto a, auto& r){ return RandomFlip{}.p(0.5f)(std::move(a), r); })
        .add([](auto a, auto& r){ return RandomRotate{}.range(-15.f, 15.f)(std::move(a), r); })
        .add([](auto a, auto& r){ return RandomCrop{}.width(600).height(460)(std::move(a), r); })
        .add([](auto a, auto& r){ return RandomZoom{}.range(0.8f, 1.0f)(std::move(a), r); });

    auto result = pipeline(std::move(ann), rng);
    std::cout << "After augmentation: " << result.boxes.size() << " box(es)\n";
    for (const auto& b : result.boxes)
        std::cout << "  [" << b.label << "] " << b.box << "\n";

    cv::Mat mat2(480, 640, CV_8UC3, cv::Scalar(0));
    AnnotatedImage<BGR> ann2{Image<BGR>(mat2), {BBox{cv::Rect2f(10.f, 10.f, 40.f, 40.f), 0, "obj"}}};
    auto result2 = std::move(ann2) | pipeline.bind(rng);
    std::cout << "Pipeline form: " << result2.boxes.size() << " box(es)\n";

    return 0;
}
