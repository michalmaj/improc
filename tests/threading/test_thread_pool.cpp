// tests/threading/test_thread_pool.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include <atomic>
#include <chrono>
#include <thread>
#include "improc/threading/thread_pool.hpp"

using improc::threading::ThreadPool;

TEST(ThreadPoolTest, SubmitReturnsCorrectResult) {
    ThreadPool pool(2);
    auto future = pool.submit([]{ return 42; });
    EXPECT_EQ(future.get(), 42);
}

TEST(ThreadPoolTest, SubmitWithArgs) {
    ThreadPool pool(2);
    auto future = pool.submit([](int a, int b){ return a + b; }, 3, 4);
    EXPECT_EQ(future.get(), 7);
}

TEST(ThreadPoolTest, SubmitDetachedDoesNotBlock) {
    ThreadPool pool(2);
    std::atomic<bool> ran{false};
    pool.submit_detached([&ran]{
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        ran = true;
    });
    EXPECT_FALSE(ran.load());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_TRUE(ran.load());
}

TEST(ThreadPoolTest, MultipleConcurrentTasks) {
    ThreadPool pool(4);
    std::vector<std::future<int>> futures;
    for (int i = 0; i < 10; ++i)
        futures.push_back(pool.submit([i]{ return i * 2; }));
    for (int i = 0; i < 10; ++i)
        EXPECT_EQ(futures[i].get(), i * 2);
}

TEST(ThreadPoolTest, DestructorDrainsQueue) {
    std::atomic<int> count{0};
    {
        ThreadPool pool(2);
        for (int i = 0; i < 5; ++i)
            pool.submit_detached([&count]{
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                ++count;
            });
    }  // destructor must wait for all tasks
    EXPECT_EQ(count.load(), 5);
}

TEST(ThreadPoolTest, ZeroThreadsThrows) {
    EXPECT_THROW(ThreadPool(0), improc::ParameterError);
}

TEST(ThreadPoolTest, ExceptionPropagatesThroughFuture) {
    ThreadPool pool(2);
    auto future = pool.submit([]() -> int {
        throw std::runtime_error("task error");
    });
    EXPECT_THROW(future.get(), std::runtime_error);
}

TEST(ThreadPoolTest, DefaultConstructionWorks) {
    ThreadPool pool;
    auto future = pool.submit([]{ return true; });
    EXPECT_TRUE(future.get());
}
