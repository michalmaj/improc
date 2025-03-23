#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/photo.hpp>
#include <stdexcept>
#include <format>
#include <thread>
#include <atomic>
#include <shared_mutex>
#include <matplot/matplot.h>

class CameraCapture {
public:
    explicit CameraCapture(unsigned int camera_id) : camera_id_(camera_id), keep_running_(true), stopped_(false) {
        start();
    }

    ~CameraCapture() {
        stop();
    }

    CameraCapture(const CameraCapture &other) = delete;

    CameraCapture(CameraCapture &&other) = delete;

    CameraCapture &operator=(const CameraCapture &other) noexcept = delete;

    CameraCapture &operator=(CameraCapture &&other) noexcept = delete;

    void stop() {
        if (stopped_) {
            return;
        }
        stopped_ = true;

        keep_running_ = false;
        if (capture_thread_.joinable()) {
            capture_thread_.join();
        }

        if (cap_.isOpened()) {
            cap_.release();
        }
    }

    cv::Mat getFrame() {
        std::shared_lock<std::shared_mutex> lock(frame_mutex_);
        return frame_.clone();
    }

    const std::string &getWindowName() const {
        return window_name_;
    }

private:
    unsigned int camera_id_{};
    std::atomic<bool> keep_running_{};
    std::atomic<bool> stopped_;
    std::thread capture_thread_;
    cv::VideoCapture cap_;
    cv::Mat frame_;
    std::shared_mutex frame_mutex_;
    std::string window_name_{"Frame"};

    void start() {
        capture_thread_ = std::thread(&CameraCapture::captureLoop, this);
    }

    void captureLoop() {
        cap_.open(camera_id_);
        if (!cap_.isOpened()) {
            std::cerr << "Cannot open camera " << camera_id_ << std::endl;
            return;
        }
        while (keep_running_) {
            cv::Mat temp_frame;
            cap_.read(temp_frame);

            if (temp_frame.empty()) {
                continue;
            }

            std::unique_lock<std::shared_mutex> lock(frame_mutex_);
            frame_ = std::move(temp_frame);
        }
    }
};

void pencilSketch(const std::atomic<bool> &running, CameraCapture &camera, cv::Mat &output,
                  std::shared_mutex &output_mutex) {
    while (running) {
        cv::Mat frame = camera.getFrame();
        if (!frame.empty()) {
            // Convert to grayscale
            cv::Mat sketch;
            cv::cvtColor(frame, sketch, cv::COLOR_BGR2GRAY);

            // Process gray image
            cv::Mat processed = sketch.clone();
            // Invert colors
            cv::bitwise_not(processed, processed);
            // Blur image with high kernel
            cv::GaussianBlur(processed, processed, cv::Size(21, 21), 0);
            // Color Dodge Blend: S = I / (255 - B) * 255
            processed = 255 - processed;
            cv::divide(sketch, processed, sketch, 256.0);

            std::unique_lock<std::shared_mutex> lock(output_mutex);
            output = std::move(sketch);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void cartoonify(const std::atomic<bool> &running, CameraCapture &camera, cv::Mat &output,
                std::shared_mutex &output_mutex) {
    while (running) {
        cv::Mat frame = camera.getFrame();
        if (!frame.empty()) {
            // Convert frame to grayscale
            cv::Mat processed, color, cartoon;
            cv::cvtColor(frame, processed, cv::COLOR_BGR2GRAY);

            // Apply blur
            cv::medianBlur(processed, processed, 3);

            // Find edges
            cv::adaptiveThreshold(processed, processed, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 9, 9);

            // Adding drawing effect
            // cv::bilateralFilter(frame, color, 9, 100, 100);
            cv::GaussianBlur(frame, color, cv::Size(5, 5), 0);

            // Combine color image and edges
            cv::bitwise_and(color, color, cartoon, processed);

            std::unique_lock<std::shared_mutex> lock(output_mutex);
            output = std::move(cartoon);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

int main() {
    CameraCapture camera{0};

    cv::Mat sketch;
    std::atomic<bool> sketch_running{true};
    std::shared_mutex sketch_mutex;

    std::thread sketch_thread{
        pencilSketch,
        std::cref(sketch_running),
        std::ref(camera),
        std::ref(sketch),
        std::ref(sketch_mutex)
    };

    cv::Mat cartoon;
    std::atomic<bool> cartoon_running{true};
    std::shared_mutex cartoon_mutex;

    std::thread cartoon_thread(cartoonify,
                               std::cref(cartoon_running),
                               std::ref(camera),
                               std::ref(cartoon),
                               std::ref(cartoon_mutex));

    while (true) {
        cv::Mat frame = camera.getFrame();
        cv::Mat sketch_copy;
        cv::Mat cartoon_copy;

        if (!frame.empty()) {
            cv::imshow(camera.getWindowName(), frame);
        } {
            std::shared_lock<std::shared_mutex> lock(sketch_mutex);
            if (!sketch.empty()) {
                sketch_copy = sketch.clone();
            }
        }

        if (!sketch_copy.empty()) {
            cv::imshow("Sketch Pencil", sketch_copy);
        } {
            std::shared_lock<std::shared_mutex> lock(cartoon_mutex);
            if (!cartoon.empty()) {
                cartoon_copy = cartoon.clone();
            }
        }

        if (!cartoon_copy.empty()) {
            cv::imshow("Cartoonify", cartoon_copy);
        }

        auto c = cv::waitKey(1);
        if (c == 'q') {
            break;
        }
    }

    sketch_running = false;
    if (sketch_thread.joinable()) {
        sketch_thread.join();
    }

    cartoon_running = false;
    if (cartoon_thread.joinable()) {
        cartoon_thread.join();
    }

    cv::destroyAllWindows();

    return 0;
}
