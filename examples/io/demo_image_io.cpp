// examples/io/demo_image_io.cpp
//
// Demo: imread / imwrite

#include <iostream>
#include "improc/core/pipeline.hpp"
#include "improc/io/image_io.hpp"

using namespace improc::core;
using namespace improc::io;

int main() {
    // Write a test image
    cv::Mat raw(64, 64, CV_8UC3, cv::Scalar(42, 100, 200));
    Image<BGR> img(raw);

    const std::string path = "/tmp/demo_improc_io.png";

    auto write_result = imwrite(path, img);
    if (!write_result) {
        std::cerr << "Write failed: " << write_result.error().message << "\n";
        return 1;
    }
    std::cout << "Wrote " << path << "\n";

    // Read back as BGR
    auto bgr_result = imread<BGR>(path);
    if (!bgr_result) {
        std::cerr << "Read failed: " << bgr_result.error().message << "\n";
        return 1;
    }
    std::cout << "Read as BGR: " << bgr_result->cols() << "x" << bgr_result->rows() << "\n";

    // Read back as Gray
    auto gray_result = imread<Gray>(path);
    if (!gray_result) {
        std::cerr << "Read Gray failed: " << gray_result.error().message << "\n";
        return 1;
    }
    std::cout << "Read as Gray: type=" << gray_result->mat().type() << " (expect 0=CV_8UC1)\n";

    // Read back as Float32C3
    auto float_result = imread<Float32C3>(path);
    if (!float_result) {
        std::cerr << "Read Float32C3 failed: " << float_result.error().message << "\n";
        return 1;
    }
    double mn, mx;
    cv::minMaxLoc(float_result->mat(), &mn, &mx);
    std::cout << "Read as Float32C3: range [" << mn << ", " << mx << "] (expect within [0,1])\n";

    // Non-existent file
    auto bad = imread<BGR>("/nonexistent/path.png");
    std::cout << "Non-existent file error: " << (bad ? "unexpected success" : bad.error().message) << "\n";

    std::cout << "demo_image_io: OK\n";
    return 0;
}
