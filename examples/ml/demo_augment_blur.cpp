// examples/ml/demo_augment_blur.cpp
// Demo: RandomBlur, RandomSharpness
// Usage: run from build directory; press any key to advance.

#include "improc/ml/augmentation.hpp"
#include "improc/core/pipeline.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace improc::core;
using namespace improc::ml;

static void show(const std::string& title, const Image<BGR>& img) {
    cv::imshow(title, img.mat());
    cv::waitKey(0);
}

int main() {
    std::mt19937 rng(42);

    cv::Mat raw(256, 256, CV_8UC3);
    for (int r = 0; r < raw.rows; ++r)
        for (int c = 0; c < raw.cols; ++c)
            raw.at<cv::Vec3b>(r, c) = {
                static_cast<uchar>(c),
                static_cast<uchar>(r),
                static_cast<uchar>(255 - c)
            };
    Image<BGR> src(raw);
    show("0 - Source", src);

    // Gaussian only
    for (int i = 0; i < 2; ++i) {
        Image<BGR> result = RandomBlur{}
            .types({RandomBlur::Type::Gaussian})
            .kernel_size(3, 15)(src, rng);
        show("1 - RandomBlur Gaussian k=[3,15] sample " + std::to_string(i + 1), result);
    }

    // Median only
    for (int i = 0; i < 2; ++i) {
        Image<BGR> result = RandomBlur{}
            .types({RandomBlur::Type::Median})
            .kernel_size(3, 15)(src, rng);
        show("2 - RandomBlur Median k=[3,15] sample " + std::to_string(i + 1), result);
    }

    // Bilateral only
    for (int i = 0; i < 2; ++i) {
        Image<BGR> result = RandomBlur{}
            .types({RandomBlur::Type::Bilateral})
            .kernel_size(3, 9)(src, rng);
        show("3 - RandomBlur Bilateral k=[3,9] sample " + std::to_string(i + 1), result);
    }

    // All types
    for (int i = 0; i < 3; ++i) {
        Image<BGR> result = RandomBlur{}.kernel_size(3, 11)(src, rng);
        show("4 - RandomBlur All types k=[3,11] sample " + std::to_string(i + 1), result);
    }

    // RandomSharpness
    for (float strength : {0.0f, 0.5f, 1.0f, 2.0f}) {
        std::mt19937 rng_s(42);
        Image<BGR> result = RandomSharpness{}.range(strength, strength).p(1.0f)(src, rng_s);
        show("5 - RandomSharpness strength=" + std::to_string(strength), result);
    }

    // Compose blur + sharpen
    auto pipeline = Compose<BGR>{}
        .add(RandomBlur{}.types({RandomBlur::Type::Gaussian, RandomBlur::Type::Median})
                         .kernel_size(3, 7))
        .add(RandomSharpness{}.range(0.5f, 1.5f).p(0.5f));
    for (int i = 0; i < 3; ++i) {
        Image<BGR> result = pipeline(src, rng);
        show("6 - Compose(Blur+Sharpen) sample " + std::to_string(i + 1), result);
    }

    std::cout << "Done.\n";
    return 0;
}
