//
// Created by Michał Maj on 09/04/2026.
//

#include "improc/core/image.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace improc::core;

int main() {
    // --- 1. Construct Image<BGR> from cv::Mat ---
    cv::Mat raw(300, 400, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(raw);
    std::cout << "Image size: " << img.cols() << "x" << img.rows() << "\n";

    cv::imshow("Original (BGR)", img.mat());
    cv::waitKey(0);

    // --- 2. Shallow copy — shared underlying data ---
    Image<BGR> shallow = img;
    shallow.mat().at<cv::Vec3b>(0, 0) = {0, 0, 255};

    std::cout << "After shallow copy modification, original pixel (0,0): "
              << img.mat().at<cv::Vec3b>(0, 0) << "  (changed — shared data)\n";

    // --- 3. Deep copy (clone) — independent data ---
    Image<BGR> deep = img.clone();
    deep.mat().at<cv::Vec3b>(0, 0) = {0, 255, 0};

    std::cout << "After deep copy modification, original pixel (0,0): "
              << img.mat().at<cv::Vec3b>(0, 0) << "  (unchanged — independent data)\n";

    cv::imshow("Deep copy (modification visible only in clone)", deep.mat());
    cv::waitKey(0);

    // --- 4. Type validation — wrong cv::Mat type throws ---
    try {
        cv::Mat wrong(100, 100, CV_8UC1);  // single-channel, but BGR expected
        Image<BGR> bad(wrong);
    } catch (const std::invalid_argument& e) {
        std::cout << "Expected error caught: " << e.what() << "\n";
    }

    return 0;
}
