# Design: `improc::core::Image<Format>` Wrapper

**Date:** 2026-04-09  
**Status:** Approved

## Overview

A thin, type-safe wrapper over `cv::Mat` for `improc::core`. Provides compile-time format safety, ergonomic API, and pipeline composition via `operator|`. Follows `cv::Mat` shallow-copy semantics — explicit `clone()` required for deep copy.

## Goals

- Catch format mismatches (e.g. passing Gray where BGR is expected) at compile-time, not runtime
- Interop with raw OpenCV via explicit `mat()` accessor only (no implicit conversion)
- Support pipeline-style processing: `image | ToGray{} | ToFloat32{}`
- MVP format set: `BGR`, `Gray`, `BGRA`, `Float32` — extensible via traits

## Non-Goals

- Not a replacement for `cv::Mat` — raw `.mat()` is always available
- No CUDA support in this iteration
- No implicit `operator cv::Mat()` — can be added later without breaking changes

---

## Components

### 1. Format Tags (`include/improc/core/format_traits.hpp`)

Puste struktury używane jako parametry szablonu:

```cpp
namespace improc::core {
    struct BGR {};
    struct Gray {};
    struct BGRA {};
    struct Float32 {};
}
```

### 2. FormatTraits (`include/improc/core/format_traits.hpp`)

Specjalizacje definiują właściwości każdego formatu. Brak specjalizacji dla nieznanego typu = błąd kompilacji.

```cpp
template<typename Format> struct FormatTraits; // celowo niezdefiniowany

template<> struct FormatTraits<BGR>     { static constexpr int cv_type = CV_8UC3;  static constexpr int channels = 3; };
template<> struct FormatTraits<Gray>    { static constexpr int cv_type = CV_8UC1;  static constexpr int channels = 1; };
template<> struct FormatTraits<BGRA>    { static constexpr int cv_type = CV_8UC4;  static constexpr int channels = 4; };
template<> struct FormatTraits<Float32> { static constexpr int cv_type = CV_32FC1; static constexpr int channels = 1; };
```

### 3. `Image<Format>` (`include/improc/core/image.hpp`)

```cpp
template<typename Format>
class Image {
public:
    explicit Image(cv::Mat mat);   // throws std::invalid_argument if mat.type() != FormatTraits<Format>::cv_type

    Image clone() const;           // deep copy

    cv::Mat&       mat();
    const cv::Mat& mat() const;

    int  rows()  const;
    int  cols()  const;
    bool empty() const;

    // Copy/move — shallow (follows cv::Mat semantics, shared data)
    Image(const Image&)            = default;
    Image& operator=(const Image&) = default;
    Image(Image&&)                 = default;
    Image& operator=(Image&&)      = default;

private:
    cv::Mat mat_;
};
```

### 4. Conversions (`include/improc/core/convert.hpp`)

Default template deleted — only explicitly defined conversions compile:

```cpp
namespace improc::core {

template<typename To, typename From>
Image<To> convert(const Image<From>&) = delete;

// Allowed conversions (MVP):
template<> Image<Gray>    convert(const Image<BGR>&     src);
template<> Image<BGR>     convert(const Image<Gray>&    src);
template<> Image<BGRA>    convert(const Image<BGR>&     src);
template<> Image<BGR>     convert(const Image<BGRA>&    src);
template<> Image<Float32> convert(const Image<Gray>&    src);

} // namespace improc::core
```

### 5. Pipeline (`include/improc/core/pipeline.hpp`)

Single global `operator|` template + conversion functors:

```cpp
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

Usage:
```cpp
auto result = bgr_image | ToGray{} | ToFloat32{};
```

---

## Error Handling

- **Construction** — `std::invalid_argument` if `cv::Mat` type doesn't match `FormatTraits`. This is the system boundary (images come from files, cameras).
- **Conversions / pipeline hot-path** — no runtime errors; type correctness guaranteed by compiler.

---

## File Structure

```
include/improc/core/
    format_traits.hpp   — FormatTraits specializations + tag types
    image.hpp           — Image<Format> class template
    convert.hpp         — convert<To>() free function templates
    pipeline.hpp        — operator| + ToGray, ToBGR, ToFloat32 functors

src/core/
    convert.cpp         — convert<> specialization implementations
    pipeline.cpp        — functor operator() implementations

tests/core/
    test_image.cpp      — construction, clone, mat accessor, invalid_argument
    test_convert.cpp    — all valid conversions, dimensions preserved
    test_pipeline.cpp   — operator|, chained conversions
```

---

## Tests

| Test case | Expectation |
|---|---|
| `Image<BGR>` constructed from `CV_8UC3` mat | no throw |
| `Image<BGR>` constructed from `CV_8UC1` mat | throws `std::invalid_argument` |
| `clone()` — modify copy | original unchanged |
| `convert<Gray>(bgr_image)` | result type `CV_8UC1`, same dimensions |
| `convert<Float32>(gray_image)` | result type `CV_32FC1` |
| `bgr \| ToGray{} \| ToFloat32{}` | final type `Image<Float32>` |
