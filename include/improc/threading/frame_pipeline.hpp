// include/improc/threading/frame_pipeline.hpp
#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include "improc/exceptions.hpp"
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
        bool expected = false;
        // Atomically transition running_ false→true; throws if already true.
        if (!running_.compare_exchange_strong(expected, true))
            throw Exception{"FramePipeline: start() called while already running"};
        processor_ = std::move(fn);
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
        auto future = std::move(pending_.front());
        pending_.pop();
        return future.get();  // may throw, but future is already popped from queue
    }

private:
    void pollLoop() {
        Processor local_processor = processor_;  // safe: written before thread started
        while (running_) {
            auto frame_result = camera_.getFrame();
            if (!frame_result) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            cv::Mat frame = std::move(*frame_result);
            try {
                auto future = pool_.submit([local_processor, frame]{ return local_processor(frame); });
                std::lock_guard<std::mutex> lock(pending_mutex_);
                pending_.push(std::move(future));
            } catch (...) {
                // pool_.submit() can throw if pool is shutting down; exit gracefully
                running_ = false;
                break;
            }
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
