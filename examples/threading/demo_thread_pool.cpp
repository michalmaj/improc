//
// Created by Michał Maj on 12/04/2026.
//

#include "improc/threading/thread_pool.hpp"
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

using improc::threading::ThreadPool;

int main() {
    // --- 1. Basic submit → future ---
    ThreadPool pool(4);
    std::cout << "ThreadPool with 4 workers\n\n";

    auto future = pool.submit([]{ return 42; });
    std::cout << "submit([] -> 42): " << future.get() << "\n";

    // --- 2. submit with args ---
    auto sum = pool.submit([](int a, int b){ return a + b; }, 10, 32);
    std::cout << "submit(add, 10, 32): " << sum.get() << "\n";

    // --- 3. Multiple concurrent tasks ---
    std::cout << "\nSubmitting 8 tasks concurrently...\n";
    std::vector<std::future<int>> futures;
    for (int i = 0; i < 8; ++i)
        futures.push_back(pool.submit([i]{
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            return i * i;
        }));
    for (int i = 0; i < 8; ++i)
        std::cout << "  task[" << i << "] = " << futures[i].get() << "\n";

    // --- 4. submit_detached (fire-and-forget) ---
    std::cout << "\nFire-and-forget (submit_detached):\n";
    std::atomic<int> counter{0};
    for (int i = 0; i < 5; ++i)
        pool.submit_detached([&counter]{
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            ++counter;  // atomic — no cout inside worker threads to avoid concurrent writes
        });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::cout << "Counter after all detached tasks: " << counter.load() << "\n";

    // --- 5. Exception propagation ---
    std::cout << "\nException propagation:\n";
    auto bad = pool.submit([]() -> int { throw std::runtime_error("task error"); });
    try {
        bad.get();
    } catch (const std::runtime_error& e) {
        std::cout << "  Caught: " << e.what() << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
