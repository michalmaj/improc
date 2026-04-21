// benchmarks/core/bench_overhead.cpp
#include <benchmark/benchmark.h>
static void BM_placeholder(benchmark::State& state) {
    for (auto _ : state) {}
}
BENCHMARK(BM_placeholder);
