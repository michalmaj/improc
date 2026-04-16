//
// Created by Michał Maj on 16/04/2026.
//
// Demo: Image augmentation module — geometric, colour, noise, and composition ops.
//
// Usage: run from the build directory; press any key to advance between windows.

#include "improc/ml/augmentation.hpp"
#include "improc/core/pipeline.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace improc::core;
using namespace improc::ml;

// Helper: show an image with a label, wait for keypress
static void show(const std::string& title, const Image<BGR>& img) {
    cv::imshow(title, img.mat());
    cv::waitKey(0);
}

int main() {
    std::mt19937 rng(42);

    // --- Source image: a simple colourful gradient ---
    cv::Mat raw(256, 256, CV_8UC3);
    for (int r = 0; r < raw.rows; ++r)
        for (int c = 0; c < raw.cols; ++c)
            raw.at<cv::Vec3b>(r, c) = {
                static_cast<uchar>(c),
                static_cast<uchar>(r),
                static_cast<uchar>(255 - c)
            };
    Image<BGR> src(raw);
    std::cout << "Source: " << src.cols() << "x" << src.rows() << " BGR\n";
    show("0 — Source", src);

    // --- 1. Geometric augmentations ---
    std::cout << "\n--- Geometric ---\n";

    Image<BGR> flipped  = RandomFlip{}.p(1.0f)(src, rng);
    show("1a — RandomFlip (p=1, horizontal)", flipped);

    Image<BGR> rotated  = RandomRotate{}.range(-30.0f, 30.0f)(src, rng);
    show("1b — RandomRotate (±30°)", rotated);

    Image<BGR> cropped  = RandomCrop{}.width(128).height(128)(src, rng);
    show("1c — RandomCrop (128×128)", cropped);
    std::cout << "  crop output: " << cropped.cols() << "x" << cropped.rows() << "\n";

    Image<BGR> resized  = RandomResize{}.range(64, 192)(src, rng);
    show("1d — RandomResize (shorter side ∈ [64, 192])", resized);
    std::cout << "  resize output: " << resized.cols() << "x" << resized.rows() << "\n";

    // --- 2. Colour augmentations ---
    std::cout << "\n--- Colour ---\n";

    Image<BGR> brightened = RandomBrightness{}.range(1.5f, 1.5f)(src, rng);
    show("2a — RandomBrightness (×1.5)", brightened);

    Image<BGR> contrasted = RandomContrast{}.range(0.3f, 0.3f)(src, rng);
    show("2b — RandomContrast (α=0.3 — low contrast)", contrasted);

    Image<BGR> jittered = ColorJitter{}
        .brightness(0.8f, 1.2f)
        .contrast(0.8f, 1.2f)
        .saturation(0.5f, 1.5f)
        .hue(-30.0f, 30.0f)(src, rng);
    show("2c — ColorJitter (brightness+contrast+saturation+hue)", jittered);

    // --- 3. Noise augmentations ---
    std::cout << "\n--- Noise ---\n";

    Image<BGR> gaussian = RandomGaussianNoise{}.std_dev(20.0f, 20.0f)(src, rng);
    show("3a — RandomGaussianNoise (σ=20)", gaussian);

    Image<BGR> sp = RandomSaltAndPepper{}.p(0.05f)(src, rng);
    show("3b — RandomSaltAndPepper (p=0.05)", sp);

    // --- 4. Composition ops ---
    std::cout << "\n--- Composition ---\n";

    // Compose: chain multiple augmentations
    auto pipeline = Compose<BGR>{}
        .add(RandomFlip{}.p(0.5f))
        .add(RandomRotate{}.range(-15.0f, 15.0f))
        .add(RandomBrightness{}.range(0.8f, 1.2f));

    for (int i = 0; i < 3; ++i) {
        Image<BGR> result = pipeline(src, rng);
        show("4a — Compose (Flip+Rotate+Brightness) — sample " + std::to_string(i + 1), result);
    }

    // RandomApply: 50% chance to apply ColorJitter
    auto maybe_jitter = RandomApply<BGR>{ColorJitter{}, 0.5f};
    for (int i = 0; i < 4; ++i) {
        Image<BGR> result = maybe_jitter(src, rng);
        show("4b — RandomApply ColorJitter p=0.5 — sample " + std::to_string(i + 1), result);
    }

    // OneOf: randomly pick one of three noise types
    auto noise_choice = OneOf<BGR>{}
        .add(RandomGaussianNoise{}.std_dev(10.0f, 25.0f))
        .add(RandomSaltAndPepper{}.p(0.03f))
        .add(RandomSaltAndPepper{}.p(0.0f));  // identity (no noise)

    for (int i = 0; i < 3; ++i) {
        Image<BGR> result = noise_choice(src, rng);
        show("4c — OneOf (Gaussian / SaltPepper / identity) — sample " + std::to_string(i + 1), result);
    }

    // --- 5. Training-style pipeline with operator| ---
    std::cout << "\n--- Training pipeline via operator| ---\n";

    auto augmentor = Compose<BGR>{}
        .add(RandomFlip{}.p(0.5f))
        .add(RandomRotate{}.range(-10.0f, 10.0f))
        .add(RandomApply<BGR>{ColorJitter{}, 0.5f})
        .add(OneOf<BGR>{}
            .add(RandomGaussianNoise{}.std_dev(5.0f, 15.0f))
            .add(RandomSaltAndPepper{}.p(0.02f)));

    for (int i = 0; i < 4; ++i) {
        Image<BGR> result = src | augmentor.bind(rng);
        show("5 — Full training pipeline — sample " + std::to_string(i + 1), result);
    }

    std::cout << "\nDone.\n";
    return 0;
}
