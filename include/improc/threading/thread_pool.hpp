// include/improc/threading/thread_pool.hpp
#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace improc::threading {

class ThreadPool {
public:
    explicit ThreadPool(std::size_t threads = std::thread::hardware_concurrency());
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    template<typename Fn, typename... Args>
    auto submit(Fn&& fn, Args&&... args)
        -> std::future<std::invoke_result_t<Fn, Args...>>;

    template<typename Fn, typename... Args>
    void submit_detached(Fn&& fn, Args&&... args);

private:
    void workerLoop();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
};

template<typename Fn, typename... Args>
auto ThreadPool::submit(Fn&& fn, Args&&... args)
    -> std::future<std::invoke_result_t<Fn, Args...>>
{
    using R = std::invoke_result_t<Fn, Args...>;
    auto task = std::make_shared<std::packaged_task<R()>>(
        std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...)
    );
    std::future<R> future = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (stop_) throw std::runtime_error("ThreadPool: cannot submit after shutdown");
        tasks_.emplace([task]{ (*task)(); });
    }
    cv_.notify_one();
    return future;
}

template<typename Fn, typename... Args>
void ThreadPool::submit_detached(Fn&& fn, Args&&... args) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (stop_) return;
        tasks_.emplace(std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...));
    }
    cv_.notify_one();
}

} // namespace improc::threading
