# Image Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `improc::core::Image<Format>` — a compile-time type-safe wrapper over `cv::Mat` with shallow-copy semantics, explicit `clone()`, format conversion free functions, and pipeline `operator|`.

**Architecture:** Format tag types (`BGR`, `Gray`, `BGRA`, `Float32`) paired with `FormatTraits` specializations drive compile-time validation. `Image<Format>` holds a `cv::Mat` and validates its type at construction. Conversions are free function template full specializations; pipeline functors wrap them behind `operator|`.

**Tech Stack:** C++23, OpenCV 4.8 (`cv::Mat`, `cv::cvtColor`, `convertTo`), GoogleTest 1.16

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `include/improc/core/format_traits.hpp` | Format tag types + `FormatTraits` specializations |
| Create | `include/improc/core/image.hpp` | `Image<Format>` class template |
| Create | `include/improc/core/convert.hpp` | `convert<To>()` declarations |
| Create | `src/core/convert.cpp` | `convert<To>()` implementations |
| Create | `include/improc/core/pipeline.hpp` | `operator|` template + functor declarations |
| Create | `src/core/pipeline.cpp` | Functor `operator()` implementations |
| Create | `tests/core/test_image.cpp` | Tests for `Image<Format>` |
| Create | `tests/core/test_convert.cpp` | Tests for `convert<To>()` |
| Create | `tests/core/test_pipeline.cpp` | Tests for `operator|` |

CMakeLists.txt uses `GLOB_RECURSE` with `CONFIGURE_DEPENDS` — no manual changes needed.

---

## Build & Test Commands

Rebuild and run all tests:
```bash
cmake --build cmake-build-debug --target improc_tests 2>&1 | tail -3
./cmake-build-debug/improc_tests
```

Run a single suite:
```bash
./cmake-build-debug/improc_tests --gtest_filter="ImageTest.*"
```

---

## Task 1: Format tags and FormatTraits

**Files:**
- Create: `include/improc/core/format_traits.hpp`

- [ ] **Step 1: Create `format_traits.hpp`**

```cpp
// include/improc/core/format_traits.hpp
#pragma once

#include <opencv2/core.hpp>

namespace improc::core {

struct BGR     {};
struct Gray    {};
struct BGRA    {};
struct Float32 {};

template<typename Format> struct FormatTraits;  // intentionally undefined — unknown format = compile error

template<> struct FormatTraits<BGR>     { static constexpr int cv_type = CV_8UC3;  static constexpr int channels = 3; };
template<> struct FormatTraits<Gray>    { static constexpr int cv_type = CV_8UC1;  static constexpr int channels = 1; };
template<> struct FormatTraits<BGRA>    { static constexpr int cv_type = CV_8UC4;  static constexpr int channels = 4; };
template<> struct FormatTraits<Float32> { static constexpr int cv_type = CV_32FC1; static constexpr int channels = 1; };

} // namespace improc::core
```

- [ ] **Step 2: Verify it compiles (no tests needed — purely compile-time)**

```bash
cmake --build cmake-build-debug --target improc__ 2>&1 | tail -5
```

Expected: build succeeds (or at worst unchanged — no new errors).

- [ ] **Step 3: Commit**

```bash
git add include/improc/core/format_traits.hpp
git commit -m "feat(core): add format tag types and FormatTraits"
```

---

## Task 2: `Image<Format>` — construction, accessors, empty/rows/cols

**Files:**
- Create: `include/improc/core/image.hpp`
- Create: `tests/core/test_image.cpp`

- [ ] **Step 1: Write failing tests**

```cpp
// tests/core/test_image.cpp
#include <gtest/gtest.h>
#include "improc/core/image.hpp"

using namespace improc::core;

TEST(ImageTest, ConstructFromValidBGRMat) {
    cv::Mat mat(100, 100, CV_8UC3);
    EXPECT_NO_THROW(Image<BGR> img(mat));
}

TEST(ImageTest, ThrowsOnWrongType) {
    cv::Mat mat(100, 100, CV_8UC1);  // Gray mat, not BGR
    EXPECT_THROW((Image<BGR>{mat}), std::invalid_argument);
}

TEST(ImageTest, RowsColsReflectMat) {
    cv::Mat mat(50, 80, CV_8UC1);
    Image<Gray> img(mat);
    EXPECT_EQ(img.rows(), 50);
    EXPECT_EQ(img.cols(), 80);
}

TEST(ImageTest, EmptyDetected) {
    Image<BGR> img(cv::Mat(0, 0, CV_8UC3));
    EXPECT_TRUE(img.empty());
}

TEST(ImageTest, NonEmptyDetected) {
    Image<BGR> img(cv::Mat(10, 10, CV_8UC3));
    EXPECT_FALSE(img.empty());
}

TEST(ImageTest, MatAccessorReturnsSameMat) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(1, 2, 3));
    Image<BGR> img(mat);
    EXPECT_EQ(img.mat().data, mat.data);  // same underlying data
}
```

- [ ] **Step 2: Run tests — confirm they fail to compile**

```bash
cmake --build cmake-build-debug --target improc_tests 2>&1 | tail -10
```

Expected: compile error — `improc/core/image.hpp` not found.

- [ ] **Step 3: Create `image.hpp`**

```cpp
// include/improc/core/image.hpp
#pragma once

#include <opencv2/core.hpp>
#include <stdexcept>
#include <format>
#include "improc/core/format_traits.hpp"

namespace improc::core {

template<typename Format>
class Image {
public:
    explicit Image(cv::Mat mat) : mat_(std::move(mat)) {
        if (mat_.type() != FormatTraits<Format>::cv_type) {
            throw std::invalid_argument(
                std::format("Image: expected cv_type {}, got {}",
                    FormatTraits<Format>::cv_type, mat_.type()));
        }
    }

    Image clone() const { return Image(mat_.clone()); }

    cv::Mat&       mat()       { return mat_; }
    const cv::Mat& mat() const { return mat_; }

    int  rows()  const { return mat_.rows; }
    int  cols()  const { return mat_.cols; }
    bool empty() const { return mat_.empty(); }

    Image(const Image&)            = default;
    Image& operator=(const Image&) = default;
    Image(Image&&)                 = default;
    Image& operator=(Image&&)      = default;

private:
    cv::Mat mat_;
};

} // namespace improc::core
```

- [ ] **Step 4: Run tests — confirm they pass**

```bash
cmake --build cmake-build-debug --target improc_tests 2>&1 | tail -3
./cmake-build-debug/improc_tests --gtest_filter="ImageTest.*"
```

Expected:
```
[==========] 5 tests from 1 test suite ran.
[  PASSED  ] 5 tests.
```

- [ ] **Step 5: Commit**

```bash
git add include/improc/core/image.hpp tests/core/test_image.cpp
git commit -m "feat(core): add Image<Format> wrapper with compile-time type safety"
```

---

## Task 3: `Image<Format>` — clone and shallow copy semantics

**Files:**
- Modify: `tests/core/test_image.cpp` (add tests)

- [ ] **Step 1: Add clone and shallow copy tests**

Append to `tests/core/test_image.cpp`:

```cpp
TEST(ImageTest, CloneIsDeepCopy) {
    cv::Mat mat(2, 2, CV_8UC3, cv::Scalar(10, 20, 30));
    Image<BGR> original(mat);
    Image<BGR> copy = original.clone();
    copy.mat().at<cv::Vec3b>(0, 0) = {0, 0, 0};
    EXPECT_EQ(original.mat().at<cv::Vec3b>(0, 0), (cv::Vec3b{10, 20, 30}));
}

TEST(ImageTest, ShallowCopySharesData) {
    cv::Mat mat(2, 2, CV_8UC3, cv::Scalar(10, 20, 30));
    Image<BGR> original(mat);
    Image<BGR> shallow = original;          // copy constructor — shallow
    shallow.mat().at<cv::Vec3b>(0, 0) = {0, 0, 0};
    EXPECT_EQ(original.mat().at<cv::Vec3b>(0, 0), (cv::Vec3b{0, 0, 0}));  // shared data changed
}
```

- [ ] **Step 2: Run tests — confirm they pass (no impl changes needed)**

```bash
cmake --build cmake-build-debug --target improc_tests 2>&1 | tail -3
./cmake-build-debug/improc_tests --gtest_filter="ImageTest.*"
```

Expected:
```
[  PASSED  ] 7 tests.
```

- [ ] **Step 3: Commit**

```bash
git add tests/core/test_image.cpp
git commit -m "test(core): add clone and shallow copy semantics tests for Image<Format>"
```

---

## Task 4: `convert<To>()` free functions

**Files:**
- Create: `include/improc/core/convert.hpp`
- Create: `src/core/convert.cpp`
- Create: `tests/core/test_convert.cpp`

- [ ] **Step 1: Write failing tests**

```cpp
// tests/core/test_convert.cpp
#include <gtest/gtest.h>
#include "improc/core/convert.hpp"

using namespace improc::core;

TEST(ConvertTest, BGRToGrayPreservesDimensions) {
    cv::Mat mat(50, 60, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> bgr(mat);
    Image<Gray> gray = convert<Gray, BGR>(bgr);
    EXPECT_EQ(gray.mat().type(), CV_8UC1);
    EXPECT_EQ(gray.rows(), 50);
    EXPECT_EQ(gray.cols(), 60);
}

TEST(ConvertTest, GrayToBGRPreservesDimensions) {
    cv::Mat mat(50, 60, CV_8UC1, cv::Scalar(128));
    Image<Gray> gray(mat);
    Image<BGR> bgr = convert<BGR, Gray>(gray);
    EXPECT_EQ(bgr.mat().type(), CV_8UC3);
    EXPECT_EQ(bgr.rows(), 50);
    EXPECT_EQ(bgr.cols(), 60);
}

TEST(ConvertTest, BGRToBGRAAddsAlphaChannel) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(10, 20, 30));
    Image<BGR> bgr(mat);
    Image<BGRA> bgra = convert<BGRA, BGR>(bgr);
    EXPECT_EQ(bgra.mat().type(), CV_8UC4);
}

TEST(ConvertTest, BGRAToBGRDropsAlphaChannel) {
    cv::Mat mat(10, 10, CV_8UC4, cv::Scalar(10, 20, 30, 255));
    Image<BGRA> bgra(mat);
    Image<BGR> bgr = convert<BGR, BGRA>(bgra);
    EXPECT_EQ(bgr.mat().type(), CV_8UC3);
}

TEST(ConvertTest, GrayToFloat32NormalizesValues) {
    cv::Mat mat(1, 1, CV_8UC1, cv::Scalar(255));
    Image<Gray> gray(mat);
    Image<Float32> f = convert<Float32, Gray>(gray);
    EXPECT_EQ(f.mat().type(), CV_32FC1);
    EXPECT_NEAR(f.mat().at<float>(0, 0), 1.0f, 1e-5f);
}

TEST(ConvertTest, GrayToFloat32ZeroIsZero) {
    cv::Mat mat(1, 1, CV_8UC1, cv::Scalar(0));
    Image<Gray> gray(mat);
    Image<Float32> f = convert<Float32, Gray>(gray);
    EXPECT_NEAR(f.mat().at<float>(0, 0), 0.0f, 1e-5f);
}
```

- [ ] **Step 2: Run tests — confirm they fail to compile**

```bash
cmake --build cmake-build-debug --target improc_tests 2>&1 | tail -10
```

Expected: compile error — `improc/core/convert.hpp` not found.

- [ ] **Step 3: Create `convert.hpp`**

```cpp
// include/improc/core/convert.hpp
#pragma once

#include "improc/core/image.hpp"

namespace improc::core {

// Primary template: deleted — unknown conversion = compile error
template<typename To, typename From>
Image<To> convert(const Image<From>&) = delete;

// Allowed conversions (full explicit specializations)
template<> Image<Gray>    convert<Gray,    BGR> (const Image<BGR>&     src);
template<> Image<BGR>     convert<BGR,     Gray>(const Image<Gray>&    src);
template<> Image<BGRA>    convert<BGRA,    BGR> (const Image<BGR>&     src);
template<> Image<BGR>     convert<BGR,     BGRA>(const Image<BGRA>&    src);
template<> Image<Float32> convert<Float32, Gray>(const Image<Gray>&    src);

} // namespace improc::core
```

- [ ] **Step 4: Create `convert.cpp`**

```cpp
// src/core/convert.cpp
#include "improc/core/convert.hpp"
#include <opencv2/imgproc.hpp>

namespace improc::core {

template<>
Image<Gray> convert<Gray, BGR>(const Image<BGR>& src) {
    cv::Mat dst;
    cv::cvtColor(src.mat(), dst, cv::COLOR_BGR2GRAY);
    return Image<Gray>(dst);
}

template<>
Image<BGR> convert<BGR, Gray>(const Image<Gray>& src) {
    cv::Mat dst;
    cv::cvtColor(src.mat(), dst, cv::COLOR_GRAY2BGR);
    return Image<BGR>(dst);
}

template<>
Image<BGRA> convert<BGRA, BGR>(const Image<BGR>& src) {
    cv::Mat dst;
    cv::cvtColor(src.mat(), dst, cv::COLOR_BGR2BGRA);
    return Image<BGRA>(dst);
}

template<>
Image<BGR> convert<BGR, BGRA>(const Image<BGRA>& src) {
    cv::Mat dst;
    cv::cvtColor(src.mat(), dst, cv::COLOR_BGRA2BGR);
    return Image<BGR>(dst);
}

template<>
Image<Float32> convert<Float32, Gray>(const Image<Gray>& src) {
    cv::Mat dst;
    src.mat().convertTo(dst, CV_32FC1, 1.0 / 255.0);
    return Image<Float32>(dst);
}

} // namespace improc::core
```

- [ ] **Step 5: Run tests — confirm they pass**

```bash
cmake --build cmake-build-debug --target improc_tests 2>&1 | tail -3
./cmake-build-debug/improc_tests --gtest_filter="ConvertTest.*"
```

Expected:
```
[==========] 6 tests from 1 test suite ran.
[  PASSED  ] 6 tests.
```

- [ ] **Step 6: Commit**

```bash
git add include/improc/core/convert.hpp src/core/convert.cpp tests/core/test_convert.cpp
git commit -m "feat(core): add convert<To,From>() free functions for format conversion"
```

---

## Task 5: Pipeline `operator|` and functors

**Files:**
- Create: `include/improc/core/pipeline.hpp`
- Create: `src/core/pipeline.cpp`
- Create: `tests/core/test_pipeline.cpp`

- [ ] **Step 1: Write failing tests**

```cpp
// tests/core/test_pipeline.cpp
#include <gtest/gtest.h>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

TEST(PipelineTest, BGRToGrayViaFunctor) {
    cv::Mat mat(50, 50, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> bgr(mat);
    Image<Gray> gray = bgr | ToGray{};
    EXPECT_EQ(gray.mat().type(), CV_8UC1);
    EXPECT_EQ(gray.rows(), 50);
}

TEST(PipelineTest, GrayToBGRViaFunctor) {
    cv::Mat mat(50, 50, CV_8UC1, cv::Scalar(128));
    Image<Gray> gray(mat);
    Image<BGR> bgr = gray | ToBGR{};
    EXPECT_EQ(bgr.mat().type(), CV_8UC3);
}

TEST(PipelineTest, GrayToFloat32ViaFunctor) {
    cv::Mat mat(1, 1, CV_8UC1, cv::Scalar(255));
    Image<Gray> gray(mat);
    Image<Float32> f = gray | ToFloat32{};
    EXPECT_EQ(f.mat().type(), CV_32FC1);
    EXPECT_NEAR(f.mat().at<float>(0, 0), 1.0f, 1e-5f);
}

TEST(PipelineTest, ChainedBGRToGrayToFloat32) {
    cv::Mat mat(50, 60, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> bgr(mat);
    Image<Float32> result = bgr | ToGray{} | ToFloat32{};
    EXPECT_EQ(result.mat().type(), CV_32FC1);
    EXPECT_EQ(result.rows(), 50);
    EXPECT_EQ(result.cols(), 60);
}
```

- [ ] **Step 2: Run tests — confirm they fail to compile**

```bash
cmake --build cmake-build-debug --target improc_tests 2>&1 | tail -10
```

Expected: compile error — `improc/core/pipeline.hpp` not found.

- [ ] **Step 3: Create `pipeline.hpp`**

```cpp
// include/improc/core/pipeline.hpp
#pragma once

#include "improc/core/image.hpp"
#include "improc/core/convert.hpp"

namespace improc::core {

template<typename Format, typename Op>
auto operator|(Image<Format> img, Op&& op) {
    return std::forward<Op>(op)(std::move(img));
}

struct ToGray    { Image<Gray>    operator()(Image<BGR>  img) const; };
struct ToBGR     { Image<BGR>     operator()(Image<Gray> img) const; };
struct ToFloat32 { Image<Float32> operator()(Image<Gray> img) const; };

} // namespace improc::core
```

- [ ] **Step 4: Create `pipeline.cpp`**

```cpp
// src/core/pipeline.cpp
#include "improc/core/pipeline.hpp"

namespace improc::core {

Image<Gray>    ToGray::operator()(Image<BGR>  img) const { return convert<Gray,    BGR> (img); }
Image<BGR>     ToBGR::operator()(Image<Gray>  img) const { return convert<BGR,     Gray>(img); }
Image<Float32> ToFloat32::operator()(Image<Gray> img) const { return convert<Float32, Gray>(img); }

} // namespace improc::core
```

- [ ] **Step 5: Run tests — confirm they pass**

```bash
cmake --build cmake-build-debug --target improc_tests 2>&1 | tail -3
./cmake-build-debug/improc_tests --gtest_filter="PipelineTest.*"
```

Expected:
```
[==========] 4 tests from 1 test suite ran.
[  PASSED  ] 4 tests.
```

- [ ] **Step 6: Run full test suite**

```bash
./cmake-build-debug/improc_tests
```

Expected:
```
[  PASSED  ] 17 tests.
```

- [ ] **Step 7: Commit**

```bash
git add include/improc/core/pipeline.hpp src/core/pipeline.cpp tests/core/test_pipeline.cpp
git commit -m "feat(core): add pipeline operator| and ToGray/ToBGR/ToFloat32 functors"
```
