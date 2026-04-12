// include/improc/threading/frame_pipeline.hpp
#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <thread>
#include <opencv2/core.hpp>
#include "improc/io/camera_capture.hpp"
#include "improc/threading/thread_pool.hpp"

namespace improc::threading {

template<typename Result>
class FramePipeline {
public:
    using Processor = std::function<Result(cv::Mat)>;

    FramePipeline(improc::io::CameraCapture& camera, ThreadPool& pool)
        : camera_(camera), pool_(pool) {}

    ~FramePipeline() { stop(); }

    FramePipeline(const FramePipeline&) = delete;
    FramePipeline& operator=(const FramePipeline&) = delete;
    FramePipeline(FramePipeline&&) = delete;
    FramePipeline& operator=(FramePipeline&&) = delete;

    void start(Processor fn) {
        if (running_) throw std::logic_error("FramePipeline: already running");
        processor_ = std::move(fn);
        running_ = true;
        poller_ = std::thread(&FramePipeline<Result>::pollLoop, this);
    }

    void stop() {
        if (!running_) return;
        running_ = false;
        if (poller_.joinable()) poller_.join();
        std::lock_guard<std::mutex> lock(pending_mutex_);
        while (!pending_.empty()) pending_.pop();
    }

    std::optional<Result> tryPop() {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        if (pending_.empty()) return std::nullopt;
        if (pending_.front().wait_for(std::chrono::seconds(0)) != std::future_status::ready)
            return std::nullopt;
        Result result = pending_.front().get();
        pending_.pop();
        return result;
    }

private:
    void pollLoop() {
        while (running_) {
            cv::Mat frame = camera_.getFrame();
            if (frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            auto future = pool_.submit([this, frame]{ return processor_(frame); });
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_.push(std::move(future));
        }
    }

    improc::io::CameraCapture& camera_;
    ThreadPool& pool_;
    Processor processor_;

    std::queue<std::future<Result>> pending_;
    std::mutex pending_mutex_;

    std::thread poller_;
    std::atomic<bool> running_{false};
};

} // namespace improc::threading
