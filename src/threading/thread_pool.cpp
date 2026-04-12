// src/threading/thread_pool.cpp
#include "improc/threading/thread_pool.hpp"

namespace improc::threading {

ThreadPool::ThreadPool(std::size_t threads) {
    if (threads == 0)
        throw std::invalid_argument("ThreadPool: thread count must be > 0");
    workers_.reserve(threads);
    try {
        for (std::size_t i = 0; i < threads; ++i)
            workers_.emplace_back(&ThreadPool::workerLoop, this);
    } catch (...) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_)
            if (w.joinable()) w.join();
        throw;
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
}

void ThreadPool::workerLoop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this]{ return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
    }
}

} // namespace improc::threading
