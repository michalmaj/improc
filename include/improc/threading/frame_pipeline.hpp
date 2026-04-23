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

/**
 * @brief Threaded frame processing pipeline for real-time camera workflows.
 *
 * Holds references (not ownership) to a `CameraCapture` and a `ThreadPool`.
 * Call `start(processor)` to begin submitting frames asynchronously.
 * `tryPop()` returns `std::optional<Result>` with the next finished result.
 *
 * @tparam Result  The return type of the processor callable.
 *
 * @code
 * FramePipeline<cv::Mat> pipeline(cam, pool);
 * pipeline.start([](cv::Mat f){ return f; });
 * while (auto r = pipeline.tryPop()) { ... }
 * @endcode
 */
template<typename Result>
class FramePipeline {
public:
    using Processor = std::function<Result(cv::Mat)>;

    /// @brief Constructs the pipeline holding references to `camera` and `pool`.
    FramePipeline(improc::io::CameraCapture& camera, ThreadPool& pool)
        : camera_(camera), pool_(pool) {}

    /// @brief Stops the pipeline and joins the polling thread.
    ~FramePipeline() { stop(); }

    /// @brief Deleted copy constructor — non-copyable.
    FramePipeline(const FramePipeline&) = delete;
    /// @brief Deleted copy assignment — non-copyable.
    FramePipeline& operator=(const FramePipeline&) = delete;
    /// @brief Deleted move constructor — non-movable.
    FramePipeline(FramePipeline&&) = delete;
    /// @brief Deleted move assignment — non-movable.
    FramePipeline& operator=(FramePipeline&&) = delete;

    /// @brief Starts the polling thread, submitting frames to the pool via `fn`.
    /// @throws improc::Exception if the pipeline is already running.
    void start(Processor fn) {
        bool expected = false;
        // Atomically transition running_ false→true; throws if already true.
        if (!running_.compare_exchange_strong(expected, true))
            throw Exception{"FramePipeline: start() called while already running"};
        processor_ = std::move(fn);
        poller_ = std::thread(&FramePipeline<Result>::pollLoop, this);
    }

    /// @brief Signals the polling thread to stop, joins it, and discards pending futures.
    void stop() {
        if (!running_) return;
        running_ = false;
        if (poller_.joinable()) poller_.join();
        std::lock_guard<std::mutex> lock(pending_mutex_);
        while (!pending_.empty()) pending_.pop();
    }

    /// @brief Returns the next ready result, or `std::nullopt` if none is available yet.
    /// @throws Any exception thrown by the processor callable, propagated from the stored future.
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
