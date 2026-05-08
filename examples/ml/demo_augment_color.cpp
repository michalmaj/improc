// examples/ml/demo_augment_color.cpp
// Demo: RandomGrayscale, RandomSolarize, RandomPosterize, RandomEqualize
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

    Image<BGR> gray = RandomGrayscale{}.p(1.0f)(src, rng);
    show("1 — RandomGrayscale (p=1)", gray);

    for (int t : {64, 128, 192}) {
        Image<BGR> sol = RandomSolarize{}.threshold(t).p(1.0f)(src, rng);
        show("2 — RandomSolarize threshold=" + std::to_string(t), sol);
    }

    for (int b : {1, 2, 4}) {
        Image<BGR> post = RandomPosterize{}.bits(b).p(1.0f)(src, rng);
        show("3 — RandomPosterize bits=" + std::to_string(b), post);
    }

    Image<BGR> eq = RandomEqualize{}.p(1.0f)(src, rng);
    show("4 — RandomEqualize (p=1)", eq);

    auto pipeline = Compose<BGR>{}
        .add(RandomGrayscale{}.p(0.3f))
        .add(RandomSolarize{}.threshold(128).p(0.5f))
        .add(RandomPosterize{}.bits(4).p(0.5f))
        .add(RandomEqualize{}.p(0.3f));
    for (int i = 0; i < 3; ++i) {
        Image<BGR> result = pipeline(src, rng);
        show("5 — Compose sample " + std::to_string(i + 1), result);
    }

    std::cout << "Done.\n";
    return 0;
}
