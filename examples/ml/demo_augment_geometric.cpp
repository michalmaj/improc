// examples/ml/demo_augment_geometric.cpp
// Demo: RandomZoom, RandomShear, RandomPerspective
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
        Image<BGR> zoomed = RandomZoom{}.range(0.6f, 1.0f)(src, rng);
        show("1 — RandomZoom [0.6, 1.0] sample " + std::to_string(i + 1), zoomed);
    }

    for (int i = 0; i < 3; ++i) {
        Image<BGR> sheared = RandomShear{}.range(-20.0f, 20.0f)(src, rng);
        show("2 — RandomShear ±20° sample " + std::to_string(i + 1), sheared);
    }

    for (int i = 0; i < 3; ++i) {
        Image<BGR> persp = RandomPerspective{}.distortion_scale(0.5f)(src, rng);
        show("3 — RandomPerspective scale=0.5 sample " + std::to_string(i + 1), persp);
    }

    auto pipeline = Compose<BGR>{}
        .add(RandomZoom{}.range(0.7f, 1.0f))
        .add(RandomShear{}.range(-10.0f, 10.0f))
        .add(RandomPerspective{}.distortion_scale(0.3f));
    for (int i = 0; i < 3; ++i) {
        Image<BGR> result = pipeline(src, rng);
        show("4 — Compose(Zoom+Shear+Perspective) sample " + std::to_string(i + 1), result);
    }

    std::cout << "Done.\n";
    return 0;
}
