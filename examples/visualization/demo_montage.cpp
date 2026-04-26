// examples/visualization/demo_montage.cpp
//
// Demonstrates Montage: arranging a collection of BGR images into a grid.
//
// All images are generated synthetically — no external files required.
// Three layouts are shown:
//   1. Default layout (auto column count, original cell size)
//   2. Fixed 4-column grid with uniform cell size and gap
//   3. Custom background color
//
// Build:  cmake --build build --target demo_montage
// Run:    ./build/demo_montage

#include <iostream>
#include <format>
#include <vector>
#include <opencv2/imgproc.hpp>
#include "improc/visualization/montage.hpp"
#include "improc/visualization/show.hpp"
#include "improc/core/image.hpp"

using improc::core::Image;
using improc::core::BGR;
using improc::visualization::Montage;
using improc::visualization::Show;

// Generate a solid-color image with a label.
static Image<BGR> make_tile(int w, int h, cv::Scalar color, const std::string& label) {
    cv::Mat mat(h, w, CV_8UC3, color);
    cv::putText(mat, label, {4, h / 2 + 6},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1, cv::LINE_AA);
    return Image<BGR>(mat);
}

int main() {
    std::cout << "=== Montage demo ===\n\n";

    // Build a set of 9 coloured tiles (simulates a batch from a dataset or augmentation pipeline).
    const std::vector<cv::Scalar> palette = {
        {200,  50,  50}, {50, 200,  50}, { 50,  50, 200},
        {200, 200,  50}, {50, 200, 200}, {200,  50, 200},
        {150, 100,  50}, {50, 150, 100}, {100,  50, 150},
    };
    std::vector<Image<BGR>> tiles;
    for (int i = 0; i < static_cast<int>(palette.size()); ++i)
        tiles.push_back(make_tile(120, 90, palette[i], std::format("img {}", i)));

    std::cout << std::format("[1] Default layout ({} images, auto cols, original size)\n",
                             tiles.size());
    Image<BGR> grid1 = Montage{tiles}();
    std::cout << std::format("    Output: {}x{}\n", grid1.cols(), grid1.rows());
    cv::imshow("Montage — default", grid1.mat());
    cv::waitKey(0);

    std::cout << "[2] Fixed 4 cols, 128x128 cells, 6 px gap\n";
    Image<BGR> grid2 = Montage{tiles}
        .cols(4)
        .cell_size(128, 128)
        .gap(6)();
    std::cout << std::format("    Output: {}x{}\n", grid2.cols(), grid2.rows());
    cv::imshow("Montage — 4 cols + gap", grid2.mat());
    cv::waitKey(0);

    std::cout << "[3] 3 cols, dark-gray background (empty cell visible)\n";
    // Use only 7 images so the last cell in a 3-col grid is empty.
    std::vector<Image<BGR>> partial(tiles.begin(), tiles.begin() + 7);
    Image<BGR> grid3 = Montage{partial}
        .cols(3)
        .cell_size(120, 90)
        .gap(4)
        .background({40, 40, 40})();
    std::cout << std::format("    Output: {}x{}\n", grid3.cols(), grid3.rows());
    cv::imshow("Montage — empty cell", grid3.mat());
    cv::waitKey(0);

    cv::destroyAllWindows();
    std::cout << "\nDone.\n";
    return 0;
}
