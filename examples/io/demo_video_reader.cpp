// examples/io/demo_video_reader.cpp
//
// Demonstrates VideoReader: sequential frame-by-frame playback.
//
// The demo always works out of the box:
//   1. Generates a short synthetic video with VideoWriter
//   2. Reads it back with VideoReader, prints metadata, and plays it
//
// Build:  cmake --build build --target demo_video_reader
// Run:    ./build/demo_video_reader

#include <iostream>
#include <format>
#include <opencv2/imgproc.hpp>
#include "improc/io/video_reader.hpp"
#include "improc/io/video_writer.hpp"
#include "improc/visualization/show.hpp"

using improc::io::VideoReader;
using improc::io::VideoWriter;
using improc::core::Image;
using improc::core::BGR;
using improc::visualization::Show;

// Generate a short synthetic video so the demo has something to read back.
static std::string make_synthetic_video(const std::string& path) {
    constexpr int W = 480, H = 270, FPS = 25, FRAMES = 75;
    VideoWriter writer{path};
    writer.fps(FPS).size(W, H);

    for (int i = 0; i < FRAMES; ++i) {
        cv::Mat mat(H, W, CV_8UC3);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                mat.at<cv::Vec3b>(y, x) = cv::Vec3b{
                    static_cast<uchar>((x * 255) / W),
                    static_cast<uchar>((y * 255) / H),
                    static_cast<uchar>((i * 255) / FRAMES)
                };
        // Overlay frame number
        cv::putText(mat, std::format("frame {}", i), {10, 30},
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, {255, 255, 255}, 2);
        writer(Image<BGR>(mat));
    }
    return path;
}

static void play_video(const std::string& path) {
    VideoReader reader{path};

    std::cout << std::format("  File:        {}\n",   path);
    std::cout << std::format("  Resolution:  {}x{}\n", reader.width(), reader.height());
    std::cout << std::format("  FPS:         {:.1f}\n", reader.fps());
    const int n = reader.frame_count();
    if (n > 0)
        std::cout << std::format("  Frame count: {}\n", n);
    else
        std::cout << "  Frame count: unknown (container does not report it)\n";

    std::cout << "  Playing — press any key to advance, ESC to stop.\n";

    int count = 0;
    while (auto frame = reader.next()) {
        cv::imshow("VideoReader demo", frame->mat());
        if (cv::waitKey(40) == 27) break;  // ~25 fps, ESC to stop
        ++count;
    }
    cv::destroyAllWindows();
    std::cout << std::format("  Read {} frames.\n", count);
}

int main() {
    // --- 1. Synthetic video (no camera / external file required) ---
    std::cout << "=== VideoReader demo ===\n\n";
    std::cout << "[1] Generating synthetic video...\n";
    const std::string tmp = "demo_video_reader_out.mp4";
    make_synthetic_video(tmp);
    std::cout << "    Written: " << tmp << "\n\n";

    std::cout << "[2] Reading back with VideoReader:\n";
    play_video(tmp);

    // --- 2. User-provided file (optional) ---
    // To play your own video, pass its path as argv[1]:
    //   ./demo_video_reader /path/to/clip.mp4
    std::cout << "\n[3] To play your own file: ./demo_video_reader /path/to/clip.mp4\n";

    std::cout << "\nDone.\n";
    return 0;
}
