// include/improc/threading/thread_pool.hpp
#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>
#include "improc/exceptions.hpp"

namespace improc::threading {

/**
 * @brief Fixed-size thread pool for submitting callable tasks.
 *
 * `submit()` enqueues work and returns a `std::future<T>` for the result.
 * `submit_detached()` is fire-and-forget. The destructor drains the queue
 * and joins all workers. Non-copyable and non-movable.
 *
 * @code
 * ThreadPool pool(4);
 * auto fut = pool.submit([]{ return 42; });
 * int result = fut.get();
 * @endcode
 */
class ThreadPool {
public:
    /// @brief Constructs the pool and starts `threads` worker threads (defaults to hardware concurrency).
    explicit ThreadPool(std::size_t threads = std::max<std::size_t>(1, std::thread::hardware_concurrency()));
    /// @brief Drains the task queue and joins all worker threads.
    ~ThreadPool();

    /// @brief Deleted copy constructor — non-copyable.
    ThreadPool(const ThreadPool&) = delete;
    /// @brief Deleted copy assignment — non-copyable.
    ThreadPool& operator=(const ThreadPool&) = delete;
    /// @brief Deleted move constructor — non-movable.
    ThreadPool(ThreadPool&&) = delete;
    /// @brief Deleted move assignment — non-movable.
    ThreadPool& operator=(ThreadPool&&) = delete;

    /// @brief Enqueues a callable and returns a future for its result.
    /// @throws improc::Exception if the pool has already been shut down.
    template<typename Fn, typename... Args>
    auto submit(Fn&& fn, Args&&... args)
        -> std::future<std::invoke_result_t<Fn, Args...>>;

    /// @brief Enqueues a callable with no result tracking; silently dropped if pool is shutting down.
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
        [fn = std::forward<Fn>(fn), ...args = std::forward<Args>(args)]() mutable {
            return std::invoke(std::move(fn), std::move(args)...);
        }
    );
    std::future<R> future = task->get_future();
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (stop_) throw Exception{"ThreadPool: cannot submit tasks after shutdown"};
        tasks_.emplace([task]{ (*task)(); });
    }
    cv_.notify_one();
    return future;
}

template<typename Fn, typename... Args>
void ThreadPool::submit_detached(Fn&& fn, Args&&... args) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (stop_) return;  // silently drop: pool is shutting down
        tasks_.emplace([fn = std::forward<Fn>(fn), ...args = std::forward<Args>(args)]() mutable {
            std::invoke(std::move(fn), std::move(args)...);
        });
    }
    cv_.notify_one();
}

} // namespace improc::threading
