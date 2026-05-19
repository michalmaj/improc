// examples/core/demo_background_subtract.cpp
//
// Usage: ./demo_background_subtract [video_file]
// With no argument: opens webcam 0.
// Shows the foreground mask alongside the original frame.
// Press any key to quit.
//
#include "improc/io/io.hpp"
#include "improc/core/pipeline.hpp"
#include "improc/core/ops/background_subtract.hpp"
#include "improc/threading/frame_pipeline.hpp"
#include "improc/threading/thread_pool.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>

using namespace improc::io;
using namespace improc::core;
using improc::threading::FramePipeline;
using improc::threading::ThreadPool;

int main(int argc, char* argv[]) {
    BackgroundSubtractMOG2 sub;

    if (argc > 1) {
        VideoFileCapture cap(argv[1]);
        cap.start();

        ThreadPool pool(2);
        FramePipeline<cv::Mat> pipeline(cap, pool);
        pipeline.start([&sub](CameraFrame frame) -> cv::Mat {
            if (!frame.rgb) return {};
            Image<Gray> fg = *frame.rgb | sub;
            cv::Mat vis;
            cv::cvtColor(fg.mat(), vis, cv::COLOR_GRAY2BGR);
            return vis;
        });

        while (true) {
            if (auto result = pipeline.tryPop()) {
                if (!result->empty()) cv::imshow("Foreground Mask", *result);
            }
            if (cv::waitKey(30) >= 0) break;
            auto probe = cap.getFrame();
            if (!probe && probe.error().code == improc::Error::Code::EndOfFile) break;
        }
        pipeline.stop();
    } else {
        WebcamCapture cam(0);
        cam.start();
        std::cout << "Warming up webcam...\n";

        ThreadPool pool(2);
        FramePipeline<cv::Mat> pipeline(cam, pool);
        pipeline.start([&sub](CameraFrame frame) -> cv::Mat {
            if (!frame.rgb) return {};
            Image<Gray> fg = *frame.rgb | sub;
            cv::Mat vis;
            cv::cvtColor(fg.mat(), vis, cv::COLOR_GRAY2BGR);
            return vis;
        });

        std::cout << "Press any key to stop...\n";
        while (true) {
            if (auto result = pipeline.tryPop()) {
                if (!result->empty()) cv::imshow("Foreground Mask (webcam)", *result);
            }
            if (cv::waitKey(1) >= 0) break;
        }
        pipeline.stop();
    }

    cv::destroyAllWindows();
    return 0;
}
