// examples/views/demo_views_m3.cpp
// M3 demo: lazy views over external sources — directory and video.
//
// DirView:   from_dir() scans a directory and lazily loads images on iteration.
// VideoView: wraps a VideoReader and lazily reads frames one at a time.

#include <filesystem>
#include <format>
#include <iostream>
#include <vector>
#include "improc/core/pipeline.hpp"
#include "improc/io/image_io.hpp"
#include "improc/io/video_reader.hpp"
#include "improc/views/views.hpp"

namespace fs    = std::filesystem;
namespace views = improc::views;
using namespace improc::core;
using improc::io::VideoReader;

// ── Helpers ───────────────────────────────────────────────────────────────────

static fs::path make_temp_dir() {
    auto tmp = fs::temp_directory_path() / "demo_views_m3";
    fs::create_directories(tmp);
    return tmp;
}

// Write N synthetic PNG images into dir.
static void write_synthetic_images(const fs::path& dir, int n) {
    for (int i = 0; i < n; ++i) {
        int side = (i % 2 == 0) ? 128 : 64;
        cv::Mat mat(side, side, CV_8UC3, cv::Scalar(i * 20, 100, 200));
        auto path = dir / std::format("img_{:03d}.png", i);
        cv::imwrite(path.string(), mat);
    }
}

// ── DirView demo ──────────────────────────────────────────────────────────────

static void demo_dir_view(const fs::path& dir) {
    std::cout << "\n── DirView demo ───────────────────────────────────────────\n";

    // -- count all images lazily
    int total = 0;
    for (const auto& img : views::from_dir(dir, {".png"})) {
        (void)img;
        ++total;
    }
    std::cout << std::format("from_dir: {} images found\n", total);

    // -- transform: resize every image to 32×32
    auto resized = views::from_dir(dir, {".png"})
        | views::transform(Resize{}.width(32).height(32))
        | views::to<std::vector<Image<BGR>>>();

    std::cout << std::format("After transform(32×32): {} images, each {}×{}\n",
        resized.size(), resized[0].cols(), resized[0].rows());

    // -- filter: keep only large (128px) images
    auto large_only = views::from_dir(dir, {".png"})
        | views::filter([](const Image<BGR>& img) { return img.cols() == 128; })
        | views::to<std::vector<Image<BGR>>>();

    std::cout << std::format("After filter(cols==128): {} images\n", large_only.size());

    // -- take: first 3 images
    auto first_three = views::from_dir(dir, {".png"})
        | views::take(3)
        | views::to<std::vector<Image<BGR>>>();

    std::cout << std::format("After take(3): {} images\n", first_three.size());

    // -- drop: skip first 4
    auto after_drop = views::from_dir(dir, {".png"})
        | views::drop(4)
        | views::to<std::vector<Image<BGR>>>();

    std::cout << std::format("After drop(4): {} images\n", after_drop.size());

    // -- composition: filter → transform → take
    auto pipeline_result = views::from_dir(dir, {".png"})
        | views::filter([](const Image<BGR>& img) { return img.cols() == 128; })
        | views::transform(Resize{}.width(64).height(64))
        | views::take(3)
        | views::to<std::vector<Image<BGR>>>();

    std::cout << std::format("filter(large)|transform(64×64)|take(3): {} images, each {}×{}\n",
        pipeline_result.size(), pipeline_result[0].cols(), pipeline_result[0].rows());
}

// ── VideoView demo ────────────────────────────────────────────────────────────

static bool ffmpeg_available() {
    return std::system("ffmpeg -version > /dev/null 2>&1") == 0;
}

static fs::path make_test_video(const fs::path& dir, int frame_count) {
    auto path = dir / "test.mp4";
    // Write synthetic frames as individual PNGs, then encode via ffmpeg.
    auto frames_dir = dir / "frames";
    fs::create_directories(frames_dir);
    for (int i = 0; i < frame_count; ++i) {
        cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(i * 10, 100, 200));
        cv::imwrite((frames_dir / std::format("{:04d}.png", i)).string(), mat);
    }
    auto cmd = std::format(
        "ffmpeg -y -framerate 10 -i {}/%04d.png -c:v libx264 -pix_fmt yuv420p {} > /dev/null 2>&1",
        frames_dir.string(), path.string());
    std::system(cmd.c_str());
    fs::remove_all(frames_dir);
    return path;
}

static void demo_video_view(const fs::path& dir) {
    std::cout << "\n── VideoView demo ─────────────────────────────────────────\n";

    if (!ffmpeg_available()) {
        std::cout << "ffmpeg not found — skipping VideoView demo\n";
        return;
    }

    const int frame_count = 10;
    auto video_path = make_test_video(dir, frame_count);

    if (!fs::exists(video_path)) {
        std::cout << "Video encoding failed — skipping VideoView demo\n";
        return;
    }

    // -- count all frames lazily
    {
        VideoReader reader{video_path.string()};
        int total = 0;
        for (const auto& frame : views::VideoView{reader}) {
            (void)frame;
            ++total;
        }
        std::cout << std::format("VideoView: {} frames read\n", total);
    }

    // -- transform: resize every frame to 32×32, take first 5
    {
        VideoReader reader{video_path.string()};
        auto result = views::VideoView{reader}
            | views::transform(Resize{}.width(32).height(32))
            | views::take(5)
            | views::to<std::vector<Image<BGR>>>();

        std::cout << std::format("transform(32×32)|take(5): {} frames, each {}×{}\n",
            result.size(), result[0].cols(), result[0].rows());
    }

    // -- filter: keep frames where cols == 64 (all frames should qualify)
    {
        VideoReader reader{video_path.string()};
        auto result = views::VideoView{reader}
            | views::filter([](const Image<BGR>& frame) { return frame.cols() == 64; })
            | views::to<std::vector<Image<BGR>>>();

        std::cout << std::format("filter(cols==64): {} frames\n", result.size());
    }

    // -- drop first 3, take next 4
    {
        VideoReader reader{video_path.string()};
        auto result = views::VideoView{reader}
            | views::drop(3)
            | views::take(4)
            | views::to<std::vector<Image<BGR>>>();

        std::cout << std::format("drop(3)|take(4): {} frames\n", result.size());
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    auto tmp = make_temp_dir();

    write_synthetic_images(tmp, 8);
    demo_dir_view(tmp);
    demo_video_view(tmp);

    fs::remove_all(tmp);
    std::cout << "\nDone.\n";
    return 0;
}
