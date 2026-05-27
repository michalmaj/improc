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
| `core` | `DetectFAST` | 480×640 | 1.90 ms | FAST corner detector; threshold=10 |
| `calib` | `Undistort` | 480×640 | 1.89 ms | wrapper overhead ≈ 0 ns vs raw |
| `calib` | `StereoBM` | 480×640 | 3.81 ms | disparity map CV_16S |
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

| Algorithm | raw | improc++ |
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

### Overhead (480×640)

Wrapper overhead is within measurement noise for all ops — the algorithms dominate.

**Flow ops (ms)**

| Op | raw | improc++ |
|---|---|---|
| DenseFarnebackFlow | 17.0 | 17.2 |
| DenseDISFlow (UltraFast) | 0.832 | 0.837 |
| SparseLKFlow | 0.533 | 0.533 |
| PhaseCorrelate | 3.200 | 3.263 |

**Tracking ops (µs)**

| Op | raw | improc++ |
|---|---|---|
| CamShift | 12.5 | 12.5 |
| MeanShift | 5.8 | 5.7 |

### Throughput — optical flow (ms/frame)

| Op | 480×640 | 1080×1920 |
|---|---|---|
| DenseFarnebackFlow | 17.1 | 117.5 |
| DenseDISFlow UltraFast | 0.9 | 2.9 |
| DenseDISFlow Fast | 3.2 | 12.5 |
| DenseDISFlow Medium | 9.2 | 35.9 |
| SparseLKFlow | 0.5 | 1.3 |
| PhaseCorrelate | 3.3 | 26.3 |

### Throughput — tracking (µs/call)

| Op | 480×640 | 1080×1920 |
|---|---|---|
| CamShift | 31 | 246 |
| MeanShift | 25 | 215 |

</details>

<details>
<summary><strong>Math &amp; foundation ops</strong> — arithmetic · filters · channels · analysis</summary>

### Arithmetic overhead (µs, 480×640 CV_8UC3)

Wrapper overhead is within measurement noise for all ops.

| Op | raw | improc++ |
|---|---|---|
| Add | 18.7 | 18.5 |
| Subtract | 19.3 | 19.6 |
| Multiply | 19.1 | 19.2 |
| Divide | 182.5 | 181.6 |

### Arithmetic throughput (µs)

| Op | 480×640 | 1080×1920 |
|---|---|---|
| Add | 18.5 | 121.7 |
| Subtract | 19.6 | 121.0 |
| Multiply | 19.2 | 123.6 |
| Divide | 181.6 | 1,232.6 |

### Filter overhead (µs, 480×640 CV_8UC3)

| Op | raw | improc++ |
|---|---|---|
| BoxFilter 5×5 | 136.3 | 136.0 |
| Convolve 3×3 | 540.8 | 543.5 |

### Filter throughput (µs)

| Op | 480×640 | 1080×1920 |
|---|---|---|
| BoxFilter 5×5 | 136.0 | 929.4 |
| Convolve 3×3 | 543.5 | 3,646.1 |

### Gradient & conversion (µs, 480×640 CV_8UC1)

| Op | raw | improc++ |
|---|---|---|
| SobelGradient | 99.4 | 99.6 |
| ScharrGradient | 128.8 | 128.7 |
| ConvertScaleAbs | 31.7 | 32.0 |

### Channel ops overhead (µs)

| Op | 480×640 raw | 480×640 improc | 1080×1920 raw | 1080×1920 improc |
|---|---|---|---|---|
| SplitChannels | 25.4 | 25.4 | 175.9 | 175.5 |
| MergeChannels | 19.9 | 19.9 | 133.2 | 134.2 |

### Analysis ops (µs, 480×640 CV_8UC1)

| Op | raw | improc++ |
|---|---|---|
| IntegralImage | 45.0 | 44.9 |
| MinMaxLoc | 10.0 | 10.0 |
| MeanStdDev | 56.8 | 57.1 |
| CountNonZero | 9.5 | 9.5 |
| Reduce (Sum) | 52.5 | 52.2 |

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

<details>
<summary><strong>v0.9.0 Detectors (improc::core)</strong> — DetectFAST · DetectBlob · DetectMSER · DetectLines · DetectQR · DetectBarcode · face (model-gated)</summary>

All overhead times in ns at 480×640. Throughput in µs.

### Overhead (480×640)

| Op | raw | improc++ | delta |
|---|---|---|---|
| `DetectFAST` | 1927 µs | 1901 µs | ≈ 0 ns |
| `DetectBlob` | 790 µs | 798 µs | ≈ 0 ns |
| `DetectMSER` | 3670 µs | 3659 µs | ≈ 0 ns |
| `DetectLines` | 3048 µs | 3109 µs | ≈ 0 ns |
| `RecognizeFace::match` | 237 ns | 239 ns | +2 ns |

### Throughput — SD (480×640, µs)

| Op | Time |
|---|---|
| `DetectFAST` | 2549 µs |
| `DetectBlob` | 788 µs |
| `DetectMSER` | 3599 µs |
| `DetectLines` | 3112 µs |
| `DetectQR` | 5775 µs |
| `DetectBarcode` | 1355 µs |

### Throughput — HD (720×1280, µs)

| Op | Time |
|---|---|
| `DetectFAST` | 6145 µs |
| `DetectBlob` | 1405 µs |
| `DetectMSER` | 11536 µs |
| `DetectLines` | 9135 µs |

### Face (model-gated, 480×640)

| Op | Time |
|---|---|
| `DetectFaceYN` | requires `face_detection_yunet.onnx` |
| `RecognizeFace::embed` | requires `face_recognition_sface.onnx` |

</details>

<details>
<summary><strong>v0.9.0 Camera Geometry (improc::calib)</strong> — chessboard · calibration · pose · stereo · epipolar · ArUco</summary>

### Chessboard detection

| Op | SD (60px cell) | HD (90px cell) |
|---|---|---|
| `FindChessboardCorners` | 910 µs | 1267 µs |
| `FindChessboardCornersSB` | 7762 µs | 14342 µs |
| `RefineCorners` | 46 µs | 50 µs |

### Calibration (one-shot, 10 synthetic views)

| Op | Time |
|---|---|
| `CalibrateCamera` | 33.9 ms |
| `StereoCalibrate` | 2.83 ms |

### Undistort

| Op | SD (480×640) | HD (720×1280) |
|---|---|---|
| `Undistort` raw | 1872 µs | 5758 µs |
| `Undistort` improc++ | 1890 µs | 5745 µs |
| `UndistortMap` (init) | 201 µs | 445 µs |

### Pose estimation

| Op | Input | Time |
|---|---|---|
| `SolvePnP` | 6 pts | 28.6 µs |
| `SolvePnPRansac` | 20 pts, 20% outliers | 213 µs |
| `ProjectPoints` | 50 pts | 0.93 µs |
| `ProjectPoints` | 500 pts | 3.31 µs |
| `ProjectPoints` | 5000 pts | 29.1 µs |

### Stereo (SD 480×640)

| Op | Time |
|---|---|
| `StereoBM` | 3.81 ms |
| `StereoSGBM` | 21.7 ms |
| `StereoRectify` | 7.7 µs |
| `ReprojectTo3D` | 470 µs |

### Epipolar geometry

| Op | 20 pts | 200 pts |
|---|---|---|
| `FindFundamentalMat` | 36 µs | 27 µs |
| `FindEssentialMat` | 311 µs | 3858 µs |
| `TriangulatePoints` | 134 µs | 1417 µs |

| Op | Input | Time |
|---|---|---|
| `RecoverPose` | 50 pts | 147 µs |

### ArUco

| Op | Input | Time |
|---|---|---|
| `DetectAruco` raw | 400×400 scene | 174 µs |
| `DetectAruco` improc++ | 400×400 scene | 176 µs |
| `GenerateAruco` | 100×100 px | 3.0 µs |
| `GenerateAruco` | 200×200 px | 9.4 µs |
| `ArucoPose` | 1 marker | 4.2 µs |
| `CharucoBoard` | 5×7, 80px | 1.26 ms |

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

<details>
<summary><strong>v0.10.0-A — Photo + Stitching (improc::core)</strong></summary>

> Build: `cmake --build build --target improc_benchmarks`
> Run: `./build/improc_benchmarks --benchmark_filter="edge_preserving|detail_enhance|stylize|pencil|seamless|merge_hdr|tonemap|stitch"`

### Throughput — SD (480×640), Apple M4 Pro, Release

| Op | Time | Notes |
|---|---|---|
| EdgePreservingFilter | 15.7 ms | Recursive filter (default) |
| DetailEnhance | 12.8 ms | |
| Stylize | 58.6 ms | |
| PencilSketch | 14.3 ms | Returns {gray, color} |
| SeamlessClone | 10.3 ms | Normal clone mode |
| MergeHDR (Mertens) | 3.5 ms | 3 frames, 5 iterations |
| ToneMap (Reinhard) | 2.66 ms | Float32C3→BGR |
| Stitch | 1.81 ms | 2-image panorama, 5 iterations |

</details>
