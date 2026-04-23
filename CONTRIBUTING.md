# Contributing to improc++

Thank you for your interest in contributing!

## Development Setup

Follow the [Getting Started](README.md#getting-started) instructions to build the project.

## Workflow

1. Fork the repository and create a feature branch:
   ```bash
   git checkout -b feature/my-op
   ```

2. Follow the existing patterns:
   - New core ops: header-only functor in `include/improc/core/ops/<name>.hpp`
   - By-value `Image<F>` parameter in `operator()`, `std::move(dst)` in the return
   - No custom `operator|` in op headers — the generic one in `pipeline.hpp` handles dispatch
   - Register the new header in `pipeline.hpp`
   - Add tests in `tests/core/ops/test_<name>.cpp`

3. Write tests first (TDD). Run the suite:
   ```bash
   cmake --build build && ./build/improc_tests
   ```

4. Add Markdown-formatted Doxygen to your public header (see existing M3 ops for style).

5. Commit atomically with descriptive messages:
   ```bash
   git commit -m "feat(core): add MyOp with fluent .param() setter"
   ```

6. Open a pull request against `main`. All CI checks must pass.

## Code Style

- C++23; no C++17 fallbacks
- `AnyFormat` concept for format-generic ops; `BGRFormat` / `GrayFormat` for format-specific
- `improc::ParameterError` for invalid setter arguments; `improc::FormatError` for type mismatches
- No custom `operator|` overloads in op headers
- Headers are self-contained (`#pragma once`, all includes explicit)

## Adding a New Format

1. Add a tag struct to `include/improc/core/format_traits.hpp`
2. Add a `FormatTraits<NewTag>` specialization with `cv_type`, `channels`, `is_float`, `name`
3. Add `convert<>` specializations in `include/improc/core/convert.hpp` and `src/core/convert.cpp`
4. Add pipeline ops (e.g. `ToNewFormat{}`) in `include/improc/core/ops/`

## Reporting Issues

Please open a GitHub issue with a minimal reproducer.
