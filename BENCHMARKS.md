# Performance Benchmarks

Measured on **Apple M4 Pro** (12-core), single thread, Release build (`-O3`).  
All times are CPU time from [Google Benchmark](https://github.com/google/benchmark) 1.9.1.  
Raw JSON: [`benchmarks/results/2026-05-17-baseline.json`](benchmarks/results/2026-05-17-baseline.json)

To reproduce on your machine:

```bash
cmake --build build --target improc_benchmarks
./build/improc_benchmarks
```

> Results vary by hardware. The numbers below are a reference, not a guarantee.

---

## Quick Reference

One representative number per namespace. Full tables with all variants are in the sections below.

| Namespace | Operation | Input | Time | Notes |
|---|---|---|---|---|
| `core` | `GaussianBlur` | 480×640 | 82 µs | wrapper cost < 2% vs raw OpenCV |
| `core` | `CLAHE` | 480×640 | 328 µs | wrapper cost < 1% |
| `core` | `NLMeansDenoising` | 480×640 | **123 ms** | ⚠️ algorithmically slow — see API `@warning` |
| `core` | `DetectORB` | 480×640 | 6.1 ms | real-time capable |
| `core` | `DetectSIFT` | 480×640 | 13.6 ms | ⚠️ 2× slower than ORB; full pipeline 311 ms @ 1080p |
| `core` | `DenseFarnebackFlow` | 480×640 | 17.1 ms | two-frame optical flow |
| `core` | `DenseDISFlow` (UltraFast) | 480×640 | 0.9 ms | 3 presets: UltraFast/Fast/Medium |
| `core` | `Add` | 480×640 | 18.5 µs | wrapper overhead ≈0 ns vs raw |
| `core` | `SobelGradient` | 480×640 | 99.6 µs | returns CV_16S dx+dy pair |
| `ml` | `Compose` (3 ops) | 224×224 | 730 µs | ~1,370 img/s |
| `ml` | `IouTracker` | 100 dets | 17.5 µs | SortTracker: 254 µs — IouTracker is fastest when Kalman not needed |
| `views` | `transform \| take(16/256)` | 224×224 | 561 µs lazy · 9,061 µs eager | **16× lazy speedup** |
| `threading` | `FramePipeline` | 16 × 480×640 | 3.6× @ 4 threads | 89% parallel efficiency |

---

## Full Tables

<details>
<summary><strong>Core ops</strong> — raw OpenCV vs improc++ wrapper overhead</summary>

All ops measured at 480×640 unless noted. `delta = improc − raw`; negative = improc faster.

### Simple pixel ops (480×640)

| Op | raw (µs) | improc++ (µs) | delta (µs) |
|---|---|---|---|
| `Resize` | 89 | 90 | +1 |
| `GaussianBlur` | 81 | 82 | +1 |
| `GammaCorrection` | 69 | 71 | +2 |
| `ToFloat32C3` | 58 | 58 | 0 |
| `NormalizeTo` | 177 | 123 | **−54** |
| `WarpAffine` | 186 | 186 | 0 |
| `Invert` 480×640 | 14 | 16 | +2 |
| `Invert` 1080×1920 | 97 | 101 | +4 |
| `InRange` 480×640 | 115 | 110 | −5 |
| `ToLAB` 480×640 | 103 | 109 | +6 |
| `ToYCrCb` 480×640 | 52 | 61 | +9 |

`NormalizeTo` is faster than the naive raw equivalent because improc++ writes in-place via `convertTo` to the same buffer, avoiding a 600 KB heap allocation per call. See [engineering story 2](#2--ml-pipeline-overhead-17--eliminated).

### Morphology (µs)

| Op | 480×640 raw | 480×640 improc | 1080×1920 raw | 1080×1920 improc |
|---|---|---|---|---|
| `MorphOpen` | 57 | 57 | 315 | 312 |
| `MorphClose` | 56 | 57 | 312 | 312 |
| `MorphGradient` | 63 | 64 | 354 | 355 |
| `TopHat` | 63 | 63 | 352 | 353 |
| `BlackHat` | 63 | 63 | 353 | 352 |

### Heavier ops (µs)

| Op | 480×640 raw | 480×640 improc | 1080×1920 raw | 1080×1920 improc |
|---|---|---|---|---|
| `AdaptiveThreshold` | 361 | 362 | 2,436 | 2,439 |
| `HistogramEqualization` | 445 | 439 | 1,284 | 1,319 |
| `CLAHE` | 339 | 328 | — | — |
| `BilateralFilter` | 1,383 | 1,385 | — | — |
| `HarrisCorner` | 822 | 827 | 6,039 | 6,153 |
| `PyrDown` | 127 | 128 | 321 | 335 |
| `PyrUp` | 605 | 607 | 4,417 | 4,418 |
| `NLMeansDenoising` ⚠️ | 122,283 | 122,525 | — | — |

NLMeansDenoising at 1080×1920 has only 5 iterations due to its cost (~135–250 ms range); results vary significantly between runs on that size.

</details>

<details>
<summary><strong>Feature detection</strong> — ORB · SIFT · AKAZE detect, describe, match</summary>

All times in µs. `e2e` = detect + describe + match (brute-force).

### Detection (µs)

| Algorithm | 480×640 raw | 480×640 improc | 1080×1920 raw | 1080×1920 improc |
|---|---|---|---|---|
| ORB | 6,172 | 6,092 | 39,392 | 39,288 |
| SIFT ⚠️ | 13,305 | 13,615 | 89,298 | 89,477 |
| AKAZE | 8,562 | 8,589 | 49,873 | 47,602 |

### Description (µs)

| Algorithm | 480×640 raw | 480×640 improc | 1080×1920 raw | 1080×1920 improc |
|---|---|---|---|---|
| ORB | 1,305 | 1,277 | 6,115 | 6,123 |
| SIFT ⚠️ | 9,201 | 9,150 | 63,236 | 63,145 |

### Matching (µs)

| Algorithm | raw | improc |
|---|---|---|
| BruteForce | 630 | 620 |
| FLANN | 2,922 | 2,924 |

### End-to-end pipeline (detect + describe + match, µs)

| Algorithm | 480×640 | 1080×1920 |
|---|---|---|
| ORB | 15,638 | 93,384 |
| SIFT ⚠️ | 49,250 | 311,360 |

</details>

<details>
<summary><strong>Image analysis</strong> — FindContours · ConnectedComponents · DistanceTransform</summary>

Input: binary `Image<Gray>` (random noise → threshold). Times in µs.

| Op | 480×640 raw | 480×640 improc | 1080×1920 raw | 1080×1920 improc |
|---|---|---|---|---|
| `FindContours` | 993 | 906 | 6,334 | 6,289 |
| `ConnectedComponents` | 544 | 571 | 2,449 | 2,495 |
| `DistanceTransform` | 1,010 | 1,053 | 7,263 | 7,289 |

</details>

<details>
<summary><strong>Motion ops</strong> — optical flow · tracking · phase correlation</summary>

### Overhead (480×640, ns)

All ops measured at 480×640. `delta = improc − raw`. These algorithms run in the ms range; deltas within measurement noise are marked ≈0.

| Op | raw | improc++ | delta |
|---|---|---|---|
| DenseFarnebackFlow | 17,034,049 | 17,200,395 | ≈0 |
| DenseDISFlow (UltraFast) | 832,015 | 837,256 | ≈0 |
| SparseLKFlow | 533,205 | 532,865 | ≈0 |
| PhaseCorrelate | 3,199,945 | 3,262,721 | ≈0 |
| CamShift | 12,545 | 12,509 | ≈0 |
| MeanShift | 5,751 | 5,743 | ≈0 |

### Throughput (ms/frame)

| Op | 480×640 | 1080×1920 |
|---|---|---|
| DenseFarnebackFlow | 17.1 | 117.5 |
| DenseDISFlow UltraFast | 0.9 | 2.9 |
| DenseDISFlow Fast | 3.2 | 12.5 |
| DenseDISFlow Medium | 9.2 | 35.9 |
| SparseLKFlow | 0.5 | 1.3 |
| PhaseCorrelate | 3.3 | 26.3 |
| CamShift | 0.031 | 0.246 |
| MeanShift | 0.025 | 0.215 |

</details>

<details>
<summary><strong>Math &amp; foundation ops</strong> — arithmetic · filters · channels · analysis</summary>

### Arithmetic overhead (ns, 480×640 CV_8UC3)

`delta = improc − raw`. All deltas are within measurement noise.

| Op | raw | improc++ | delta |
|---|---|---|---|
| Add | 18,652 | 18,546 | ≈0 |
| Subtract | 19,299 | 19,575 | ≈0 |
| Multiply | 19,059 | 19,216 | ≈0 |
| Divide | 182,479 | 181,589 | ≈0 |

### Arithmetic throughput (µs)

| Op | 480×640 | 1080×1920 |
|---|---|---|
| Add | 18.5 | 121.7 |
| Subtract | 19.6 | 121.0 |
| Multiply | 19.2 | 123.6 |
| Divide | 181.6 | 1,232.6 |

### Filter overhead (ns, 480×640 CV_8UC3)

| Op | raw | improc++ | delta |
|---|---|---|---|
| BoxFilter 5×5 | 136,308 | 135,986 | ≈0 |
| Convolve 3×3 | 540,794 | 543,503 | ≈0 |

### Filter throughput (µs)

| Op | 480×640 | 1080×1920 |
|---|---|---|
| BoxFilter 5×5 | 136.0 | 929.4 |
| Convolve 3×3 | 543.5 | 3,646.1 |

### Gradient & conversion (ns, 480×640 CV_8UC1)

| Op | raw | improc++ | delta |
|---|---|---|---|
| SobelGradient | 99,392 | 99,580 | ≈0 |
| ScharrGradient | 128,848 | 128,685 | ≈0 |
| ConvertScaleAbs | 31,682 | 31,972 | ≈0 |

### Channel ops overhead (ns)

| Op | 480×640 raw | 480×640 improc++ | delta | 1080×1920 raw | 1080×1920 improc++ | delta |
|---|---|---|---|---|---|---|
| SplitChannels | 25,428 | 25,404 | ≈0 | 175,936 | 175,509 | ≈0 |
| MergeChannels | 19,878 | 19,941 | ≈0 | 133,212 | 134,153 | ≈0 |

### Analysis ops (ns, 480×640 CV_8UC1)

| Op | raw | improc++ | delta |
|---|---|---|---|
| IntegralImage | 45,026 | 44,897 | ≈0 |
| MinMaxLoc | 10,043 | 9,993 | ≈0 |
| MeanStdDev | 56,819 | 57,116 | ≈0 |
| CountNonZero | 9,467 | 9,450 | ≈0 |
| Reduce (Sum) | 52,521 | 52,215 | ≈0 |

</details>

<details>
<summary><strong>ML pipeline</strong> — Resize → CLAHE → GaussianBlur → ToFloat32C3 → NormalizeTo, 1080×1920 input</summary>

| Implementation | Time (ms) | Throughput (img/s) |
|---|---|---|
| improc++ pipeline | 0.453 | 2,207 |
| Raw OpenCV (equivalent) | 0.454 | 2,200 |

improc++ pipeline is on par with hand-written OpenCV code under equivalent allocation conditions.

</details>

<details>
<summary><strong>Augmentation</strong> — improc::ml::augment ops, single image, with RNG</summary>

Times in µs. All ops take `Image<BGR>` and an `std::mt19937` rng.

| Op | 224×224 | 640×640 |
|---|---|---|
| `RandomFlip` | 7 | 60 |
| `RandomRotate` | 143 | 209 |
| `ColorJitter` | 376 | 2,363 |
| `RandomGaussianNoise` | 361 | 2,959 |
| `BBoxCompose` | 149 | 283 |
| `RandomApply` | 4 | 30 |
| `MixUp` | 23 | 183 |
| `CutMix` | 4 | 28 |
| `Compose` (RandomFlip + ColorJitter + GaussianNoise) | 730 | 5,377 |

</details>

<details>
<summary><strong>Eval</strong> — ClassEval · DetectionEval · SegEval update() throughput</summary>

| Evaluator | Variant | Time |
|---|---|---|
| `ClassEval::update()` | — | 1.5 ns |
| `DetectionEval::update()` | 5 dets | 2.9 µs |
| `DetectionEval::update()` | 20 dets | 16.1 µs |
| `DetectionEval::update()` | 50 dets | 71.8 µs |
| `SegEval::update()` | 224×224 mask | 20.3 µs |
| `SegEval::update()` | 640×640 mask | 159.3 µs |

</details>

<details>
<summary><strong>Tracking</strong> — IouTracker · SortTracker · ByteTracker per-frame update()</summary>

Times in µs per `update()` call. All trackers are drop-in replaceable.

| Tracker | 10 dets | 50 dets | 100 dets |
|---|---|---|---|
| `IouTracker` | 0.4 | 4.6 | **17.5** |
| `SortTracker` | 22.8 | 117.7 | 253.8 |
| `ByteTracker` | 22.8 | 117.6 | 252.0 |

`IouTracker` is fastest when Kalman prediction is not needed (e.g. high-confidence detections, stable scenes). `SortTracker` and `ByteTracker` add motion prediction at ~14× the cost at 100 detections.

</details>

<details>
<summary><strong>Views &amp; Threading</strong> — lazy evaluation speedup + ThreadPool throughput</summary>

### Lazy views — `take(N)` deferred evaluation

| Variant | 224×224 | 640×640 |
|---|---|---|
| `transform \| take(16/256)` lazy | 561 µs | 1,939 µs |
| `transform \| take(16/256)` eager (all 256) | 9,061 µs | 30,179 µs |
| **Speedup** | **16×** | **16×** |

Lazy views execute only the work you consume. With `take(N)`, exactly N transforms run regardless of collection size.

### Lazy views — filter before transform (mixed batch: 128 large + 128 tiny)

| Variant | 224×224 | 640×640 |
|---|---|---|
| `filter \| transform` lazy | 4,540 µs | 16,112 µs |
| `transform \| filter` eager | 4,835 µs | 16,326 µs |

Filter-before-transform avoids processing elements that will be discarded. Speedup depends on the filter selectivity; here half of images pass the filter.

### batch(8) chunking overhead

| Input size | Time |
|---|---|
| 224×224 | 5 ns |
| 640×640 | 5 ns |

### ThreadPool — frame pipeline (16 × 480×640 → Resize+GaussianBlur → 224×224)

| Mode | Wall time (µs) | Speedup |
|---|---|---|
| Sequential | 1,862 | 1.0× |
| 2 threads | 975 | 1.91× |
| 4 threads | 513 | **3.63×** |
| 12 threads | 312 | 5.97× |

### ThreadPool::submit() latency (trivial task)

| Threads | CPU time |
|---|---|
| 1 | 2.7 µs |
| 2 | 2.8 µs |
| 4 | 2.8 µs |
| 12 | 2.7 µs |

</details>

---

## Under the Hood: Three Performance Fixes

These case studies show the debugging and optimisation work behind the numbers above.
Each started with a benchmark anomaly and ended with a root-cause fix.

---

### 1 — IouTracker: 31× speedup (549 µs → 17.5 µs @ 100 detections)

**What we saw:** IouTracker was 2× slower than SortTracker at 100 detections, despite
SortTracker running a full Kalman filter. Scaling was catastrophic — 10× more detections
produced roughly 580× more work.

**Root cause:** The greedy matching loop recomputed the full D×T IoU matrix on every
iteration (up to `min(D,T)` times), and allocated a `BBox` string label on every
inner-loop pair. With D=T=100: over 1,000,000 IoU computations per frame.

**Fix:** Pre-convert detections to `BBox` once, build the IoU matrix once (O(D·T)),
collect above-threshold pairs, sort descending, assign greedily in one pass.
Complexity: O(D·T·min(D,T)) → O(D·T·log(D·T)).

| Detections | Before | After | Speedup |
|---|---|---|---|
| 10 | 942 µs | 0.4 µs | 2.4× |
| 50 | 70,369 µs | 4.6 µs | 15,000× |
| 100 | 548,729 µs | 17.5 µs | **31×** |

---

### 2 — ML pipeline overhead: 17% → eliminated

**What we saw:** The improc++ pipeline (Resize → CLAHE → GaussianBlur → ToFloat32C3 →
NormalizeTo) measured 17% slower than equivalent hand-written OpenCV code —
1,752 img/s vs 2,097 img/s.

**Root cause — two issues found:**

**(a)** `NormalizeTo::operator()` allocated a fresh 600 KB `cv::Mat` output buffer on
every call. Since the input is already `Float32C3` and `convertTo` is safe in-place
for same-type conversions, this allocation was unnecessary.

**(b)** The raw baseline declared its intermediate `cv::Mat` objects *outside* the
benchmark loop. OpenCV detects the pre-allocated buffer and reuses it — zero heap
allocations after the first iteration. This gave raw an artificial advantage not
present in real production code.

**Fix:** (a) Write `convertTo` to `img.mat()` in-place and return `img` by move.
(b) Move raw benchmark Mat declarations inside the loop for a fair comparison.

| Implementation | Before | After |
|---|---|---|
| improc++ | 0.571 ms · 1,752 img/s | 0.453 ms · 2,207 img/s |
| Raw OpenCV | 0.477 ms · 2,097 img/s | 0.454 ms · 2,200 img/s |

improc++ is now on par with the raw baseline under equivalent conditions.

---

### 3 — Views lazy evaluation: 16× speedup demonstrated

**What we saw:** `transform | to<vector>()` and a hand-written `for` loop measured
255 µs and 258 µs respectively — 1% difference, effectively zero. Lazy views appeared
to provide no benefit.

**Root cause:** The benchmark processed all 32 images in both paths unconditionally.
Lazy evaluation only defers work when you don't consume the full range — with a plain
`to<vector>()` materialising everything, both paths do identical work.

**Fix:** Redesigned the benchmark with `take(16)` on a 256-image collection. Lazy
processes 16 images; eager processes all 256 before discarding. The implementation
was correct all along — the benchmark was the issue.

| Variant | 224×224 | 640×640 |
|---|---|---|
| `transform \| take(16/256)` **lazy** | 561 µs | 1,939 µs |
| `transform \| take(16/256)` **eager** | 9,061 µs | 30,179 µs |
| Speedup | **16×** | **16×** |
