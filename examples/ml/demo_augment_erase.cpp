// examples/ml/demo_augment_erase.cpp
// Demo: RandomErasing, GridDropout
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
    show("0 — Source", src);

    for (int i = 0; i < 3; ++i) {
        Image<BGR> erased = RandomErasing{}.p(1.0f)(src, rng);
        show("1 — RandomErasing (p=1) sample " + std::to_string(i + 1), erased);
    }

    for (int cell : {8, 16, 32}) {
        Image<BGR> dropped = GridDropout{}.ratio(0.5f).unit_size(cell)(src, rng);
        show("2 — GridDropout ratio=0.5 cell=" + std::to_string(cell), dropped);
    }

    auto pipeline = Compose<BGR>{}
        .add(RandomErasing{}.p(0.5f).value(128))
        .add(GridDropout{}.ratio(0.3f).unit_size(16));
    for (int i = 0; i < 3; ++i) {
        Image<BGR> result = pipeline(src, rng);
        show("3 — Compose sample " + std::to_string(i + 1), result);
    }

    std::cout << "Done.\n";
    return 0;
}
