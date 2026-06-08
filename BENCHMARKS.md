# Performance Benchmarks

Measured on **Apple M4 Pro** (12-core), single thread, Release build (`-O3`).  
All times are CPU time from [Google Benchmark](https://github.com/google/benchmark) 1.9.1.  
Raw JSON: [`benchmarks/results/2026-05-31-overhead.json`](benchmarks/results/2026-05-31-overhead.json)

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
| `core` | `GaussianBlur` | 480×640 | 88 µs | wrapper cost < 2% vs raw OpenCV |
| `core` | `CLAHE` | 480×640 | 441 µs | wrapper cost < 4% |
| `core` | `NLMeansDenoising` | 480×640 | **115 ms** | ⚠️ algorithmically slow — see API `@warning` |
| `core` | `DetectORB` | 480×640 | 6.5 ms | real-time capable |
| `core` | `DetectSIFT` | 480×640 | 14.3 ms | ⚠️ 2× slower than ORB; full pipeline 316 ms @ 1080p |
| `core` | `DenseFarnebackFlow` | 480×640 | 16.7 ms | two-frame optical flow |
| `core` | `DenseDISFlow` (UltraFast) | 480×640 | 0.9 ms | 3 presets: UltraFast/Fast/Medium |
| `core` | `Add` | 480×640 | 23.3 µs | wrapper overhead ≈0 ns vs raw |
| `core` | `SobelGradient` | 480×640 | 98.3 µs | returns CV_16S dx+dy pair |
| `core` | `DetectFAST` | 480×640 | 1.9 ms | FAST corner detector; threshold=10 |
| `calib` | `Undistort` | 480×640 | 1.9 ms | wrapper overhead ≈ 0 ns vs raw |
| `calib` | `StereoBM` | 480×640 | 3.8 ms | disparity map CV_16S |
| `ml` | `Compose` (3 ops) | 224×224 | 728 µs | ~1,373 img/s |
| `ml` | `IouTracker` | 100 dets | 17.1 µs | SortTracker: 256 µs — IouTracker is fastest when Kalman not needed |
| `views` | `transform \| take(16/256)` | 224×224 | 591 µs lazy · 9,500 µs eager | **16× lazy speedup** |
| `threading` | `FramePipeline` | 16 × 480×640 | 3.7× @ 4 threads | 88% parallel efficiency |

---

## Full Tables

<details>
<summary><strong>Core ops</strong> — raw OpenCV vs improc++ wrapper overhead</summary>

All ops measured at 480×640 unless noted. `delta = improc − raw`; negative = improc faster.

### Simple pixel ops (480×640)

| Op | raw (µs) | improc++ (µs) | delta (µs) |
|---|---|---|---|
| `Resize` | 90 | 91 | +1 |
| `GaussianBlur` | 87 | 88 | +1 |
| `GammaCorrection` | 59 | 60 | +1 |
| `ToFloat32C3` | 58 | 58 | 0 |
| `NormalizeTo` | 173 | 123 | **−50** |
| `WarpAffine` | 195 | 195 | 0 |
| `Invert` 480×640 | 14 | 15 | +1 |
| `Invert` 1080×1920 | 98 | 98 | 0 |
| `InRange` 480×640 | 112 | 112 | 0 |
| `ToLAB` 480×640 | 112 | 111 | −1 |
| `ToYCrCb` 480×640 | 58 | 58 | 0 |

`NormalizeTo` is faster than the naive raw equivalent because improc++ writes in-place via `convertTo` to the same buffer, avoiding a 600 KB heap allocation per call. See [engineering story 2](#2--ml-pipeline-overhead-17--eliminated).

### Morphology (µs)

| Op | 480×640 raw | 480×640 improc | 1080×1920 raw | 1080×1920 improc |
|---|---|---|---|---|
| `MorphOpen` | 56 | 56 | 309 | 311 |
| `MorphClose` | 56 | 56 | 307 | 305 |
| `MorphGradient` | 64 | 65 | 349 | 350 |
| `TopHat` | 63 | 64 | 347 | 352 |
| `BlackHat` | 63 | 63 | 357 | 352 |

### Heavier ops (µs)

| Op | 480×640 raw | 480×640 improc | 1080×1920 raw | 1080×1920 improc |
|---|---|---|---|---|
| `AdaptiveThreshold` | 358 | 359 | 2,440 | 2,440 |
| `HistogramEqualization` | 524 | 505 | 1,534 | 1,481 |
| `CLAHE` | 427 | 441 | — | — |
| `BilateralFilter` | 1,400 | 1,400 | — | — |
| `HarrisCorner` | 816 | 830 | 6,100 | 6,100 |
| `PyrDown` | 150 | 150 | 355 | 358 |
| `PyrUp` | 602 | 603 | 4,500 | 4,500 |
| `NLMeansDenoising` ⚠️ | 117,200 | 115,400 | — | — |

NLMeansDenoising at 1080×1920 has only 5 iterations due to its cost (~135–250 ms range); results vary significantly between runs on that size.

</details>

<details>
<summary><strong>Feature detection</strong> — ORB · SIFT · AKAZE detect, describe, match</summary>

All times in µs. `e2e` = detect + describe + match (brute-force).

### Detection (µs)

| Algorithm | 480×640 raw | 480×640 improc | 1080×1920 raw | 1080×1920 improc |
|---|---|---|---|---|
| ORB | 6,400 | 6,500 | 41,300 | 40,700 |
| SIFT ⚠️ | 13,800 | 14,300 | 93,400 | 92,500 |
| AKAZE | 9,400 | 9,300 | 53,100 | 52,800 |

### Description (µs)

| Algorithm | 480×640 raw | 480×640 improc | 1080×1920 raw | 1080×1920 improc |
|---|---|---|---|---|
| ORB | 1,400 | 1,400 | 6,500 | 6,500 |
| SIFT ⚠️ | 9,400 | 9,400 | 65,500 | 64,600 |

### Matching (µs)

| Algorithm | raw | improc++ |
|---|---|---|
| BruteForce | 659 | 690 |
| FLANN | 3,000 | 3,000 |

### End-to-end pipeline (detect + describe + match, µs)

| Algorithm | 480×640 | 1080×1920 |
|---|---|---|
| ORB | 16,600 | 97,300 |
| SIFT ⚠️ | 50,200 | 316,500 |

</details>

<details>
<summary><strong>Image analysis</strong> — FindContours · ConnectedComponents · DistanceTransform</summary>

Input: binary `Image<Gray>` (random noise → threshold). Times in µs.

| Op | 480×640 raw | 480×640 improc | 1080×1920 raw | 1080×1920 improc |
|---|---|---|---|---|
| `FindContours` | 937 | 948 | 6,700 | 6,800 |
| `ConnectedComponents` | 556 | 562 | 2,600 | 2,600 |
| `DistanceTransform` | 1,000 | 994 | 7,600 | 7,500 |

</details>

<details>
<summary><strong>Motion ops</strong> — optical flow · tracking · phase correlation</summary>

### Overhead (480×640)

Wrapper overhead is within measurement noise for all ops — the algorithms dominate.

**Flow ops (ms)**

| Op | raw | improc++ |
|---|---|---|
| DenseFarnebackFlow | 16.7 | 16.7 |
| DenseDISFlow (UltraFast) | 0.860 | 0.860 |
| SparseLKFlow | 0.521 | 0.518 |
| PhaseCorrelate | 3.200 | 3.200 |

**Tracking ops (µs)**

| Op | raw | improc++ |
|---|---|---|
| CamShift | 12.4 | 12.3 |
| MeanShift | 5.6 | 5.7 |

### Throughput — optical flow (ms/frame)

| Op | 480×640 | 1080×1920 |
|---|---|---|
| DenseFarnebackFlow | 16.7 | 115.8 |
| DenseDISFlow UltraFast | 0.9 | 3.2 |
| DenseDISFlow Fast | 3.3 | 14.9 |
| DenseDISFlow Medium | 10.2 | 40.0 |
| SparseLKFlow | 0.5 | 1.4 |
| PhaseCorrelate | 3.2 | 26.0 |

### Throughput — tracking (µs/call)

| Op | 480×640 | 1080×1920 |
|---|---|---|
| CamShift | 30 | 244 |
| MeanShift | 25 | 216 |

</details>

<details>
<summary><strong>Math &amp; foundation ops</strong> — arithmetic · filters · channels · analysis</summary>

### Arithmetic overhead (µs, 480×640 CV_8UC3)

Wrapper overhead is within measurement noise for all ops.

| Op | raw | improc++ |
|---|---|---|
| Add | 23.3 | 23.4 |
| Subtract | 23.5 | 23.5 |
| Multiply | 23.5 | 23.4 |
| Divide | 178.5 | 178.3 |

### Arithmetic throughput (µs)

| Op | 480×640 | 1080×1920 |
|---|---|---|
| Add | 23.3 | 158.4 |
| Subtract | 23.5 | 158.2 |
| Multiply | 23.4 | 152.8 |
| Divide | 178.3 | 1,200 |

### Filter overhead (µs, 480×640 CV_8UC3)

| Op | raw | improc++ |
|---|---|---|
| BoxFilter 5×5 | 134.8 | 135.4 |
| Convolve 3×3 | 540.3 | 536.0 |

### Filter throughput (µs)

| Op | 480×640 | 1080×1920 |
|---|---|---|
| BoxFilter 5×5 | 135.4 | 913.8 |
| Convolve 3×3 | 536.0 | 3,600 |

### Gradient & conversion (µs, 480×640 CV_8UC1)

| Op | raw | improc++ |
|---|---|---|
| SobelGradient | 97.8 | 98.3 |
| ScharrGradient | 127.0 | 126.9 |
| ConvertScaleAbs | 31.4 | 31.5 |

### Channel ops overhead (µs)

| Op | 480×640 raw | 480×640 improc | 1080×1920 raw | 1080×1920 improc |
|---|---|---|---|---|
| SplitChannels | 31.4 | 31.4 | 211.0 | 210.7 |
| MergeChannels | 19.8 | 19.8 | 131.9 | 132.0 |

### Analysis ops (µs, 480×640 CV_8UC1)

| Op | raw | improc++ |
|---|---|---|
| IntegralImage | 43.5 | 43.5 |
| MinMaxLoc | 9.9 | 9.9 |
| MeanStdDev | 52.7 | 52.8 |
| CountNonZero | 9.0 | 9.0 |
| Reduce (Sum) | 58.2 | 57.2 |

</details>

<details>
<summary><strong>ML pipeline</strong> — Resize → CLAHE → GaussianBlur → ToFloat32C3 → NormalizeTo, 1080×1920 input</summary>

| Implementation | Time (ms) | Throughput (img/s) |
|---|---|---|
| improc++ pipeline | 0.452 | 2,212 |
| Raw OpenCV (equivalent) | 0.456 | 2,193 |

improc++ pipeline is on par with hand-written OpenCV code under equivalent allocation conditions.

</details>

<details>
<summary><strong>Augmentation</strong> — improc::ml::augment ops, single image, with RNG</summary>

Times in µs. All ops take `Image<BGR>` and an `std::mt19937` rng.

| Op | 224×224 | 640×640 |
|---|---|---|
| `RandomFlip` | 7 | 60 |
| `RandomRotate` | 147 | 225 |
| `ColorJitter` | 336 | 2,400 |
| `RandomGaussianNoise` | 358 | 2,900 |
| `BBoxCompose` | 148 | 297 |
| `RandomApply` | 4 | 30 |
| `MixUp` | 23 | 182 |
| `CutMix` | 4 | 28 |
| `Compose` (RandomFlip + ColorJitter + GaussianNoise) | 728 | 5,400 |

</details>

<details>
<summary><strong>Eval</strong> — ClassEval · DetectionEval · SegEval update() throughput</summary>

| Evaluator | Variant | Time |
|---|---|---|
| `ClassEval::update()` | — | 1.5 ns |
| `DetectionEval::update()` | 5 dets | 2.9 µs |
| `DetectionEval::update()` | 20 dets | 15.7 µs |
| `DetectionEval::update()` | 50 dets | 69.2 µs |
| `SegEval::update()` | 224×224 mask | 20.0 µs |
| `SegEval::update()` | 640×640 mask | 157.4 µs |

</details>

<details>
<summary><strong>Tracking</strong> — IouTracker · SortTracker · ByteTracker per-frame update()</summary>

Times in µs per `update()` call. All trackers are drop-in replaceable.

| Tracker | 10 dets | 50 dets | 100 dets |
|---|---|---|---|
| `IouTracker` | 0.4 | 4.6 | **17.1** |
| `SortTracker` | 22.8 | 118.4 | 256.1 |
| `ByteTracker` | 23.1 | 118.4 | 256.4 |

`IouTracker` is fastest when Kalman prediction is not needed (e.g. high-confidence detections, stable scenes). `SortTracker` and `ByteTracker` add motion prediction at ~15× the cost at 100 detections.

</details>

<details>
<summary><strong>Views &amp; Threading</strong> — lazy evaluation speedup + ThreadPool throughput</summary>

### Lazy views — `take(N)` deferred evaluation

| Variant | 224×224 | 640×640 |
|---|---|---|
| `transform \| take(16/256)` lazy | 591 µs | 2,200 µs |
| `transform \| take(16/256)` eager (all 256) | 9,500 µs | 35,000 µs |
| **Speedup** | **16×** | **16×** |

Lazy views execute only the work you consume. With `take(N)`, exactly N transforms run regardless of collection size.

### Lazy views — filter before transform (mixed batch: 128 large + 128 tiny)

| Variant | 224×224 | 640×640 |
|---|---|---|
| `filter \| transform` lazy | 4,700 µs | 18,200 µs |
| `transform \| filter` eager | 5,100 µs | 18,000 µs |

Filter-before-transform avoids processing elements that will be discarded. Speedup depends on the filter selectivity; here half of images pass the filter.

### batch(8) chunking overhead

| Input size | Time |
|---|---|
| 224×224 | 5 ns |
| 640×640 | 5 ns |

### ThreadPool — frame pipeline (16 × 480×640 → Resize+GaussianBlur → 224×224)

| Mode | Wall time (µs) | Speedup |
|---|---|---|
| Sequential | 1,900 | 1.0× |
| 2 threads | 971 | 1.96× |
| 4 threads | 510 | **3.73×** |
| 12 threads | 306 | 6.21× |

### ThreadPool::submit() latency (trivial task)

| Threads | CPU time |
|---|---|
| 1 | 5.2 µs |
| 2 | 5.1 µs |
| 4 | 5.3 µs |
| 12 | 5.3 µs |

</details>

<details>
<summary><strong>Detectors (improc::core)</strong> — DetectFAST · DetectBlob · DetectMSER · DetectLines · DetectQR · DetectBarcode · face (model-gated)</summary>

All overhead times at 480×640. Throughput in µs.

### Overhead (480×640)

| Op | raw | improc++ | delta |
|---|---|---|---|
| `DetectFAST` | 1,900 µs | 1,900 µs | ≈ 0 ns |
| `DetectBlob` | 853 µs | 856 µs | ≈ 0 ns |
| `DetectMSER` | 3,800 µs | 3,800 µs | ≈ 0 ns |
| `DetectLines` | 3,200 µs | 3,200 µs | ≈ 0 ns |
| `RecognizeFace::match` | 229 ns | 230 ns | +1 ns |

### Throughput — SD (480×640, µs)

| Op | Time |
|---|---|
| `DetectFAST` | 2,100 µs |
| `DetectBlob` | 848 µs |
| `DetectMSER` | 3,800 µs |
| `DetectLines` | 3,200 µs |
| `DetectQR` | 6,300 µs |
| `DetectBarcode` | 1,500 µs |

### Throughput — HD (720×1280, µs)

| Op | Time |
|---|---|
| `DetectFAST` | 6,300 µs |
| `DetectBlob` | 1,500 µs |
| `DetectMSER` | 12,300 µs |
| `DetectLines` | 9,800 µs |

### Face (model-gated, 480×640)

| Op | Time |
|---|---|
| `DetectFaceYN` | requires `face_detection_yunet.onnx` |
| `RecognizeFace::embed` | requires `face_recognition_sface.onnx` |

</details>

<details>
<summary><strong>Camera Geometry (improc::calib)</strong> — chessboard · calibration · pose · stereo · epipolar · ArUco</summary>

### Chessboard detection

| Op | SD (60px cell) | HD (90px cell) |
|---|---|---|
| `FindChessboardCorners` | 2,087 µs | 1,266 µs |
| `FindChessboardCornersSB` | 9,384 µs | 16,212 µs |
| `RefineCorners` | 47 µs | 47 µs |

### Calibration (one-shot, 10 synthetic views)

| Op | Time |
|---|---|
| `CalibrateCamera` | 34.7 ms |
| `StereoCalibrate` | 2.86 ms |

### Undistort

| Op | SD (480×640) | HD (720×1280) |
|---|---|---|
| `Undistort` raw | 1,917 µs | 6,197 µs |
| `Undistort` improc++ | 1,924 µs | 6,191 µs |
| `UndistortMap` (init) | 225 µs | 465 µs |

### Pose estimation

| Op | Input | Time |
|---|---|---|
| `SolvePnP` | 6 pts | 31 µs |
| `SolvePnPRansac` | 20 pts, 20% outliers | 209 µs |
| `ProjectPoints` | 50 pts | 1 µs |
| `ProjectPoints` | 500 pts | 3 µs |
| `ProjectPoints` | 5000 pts | 29 µs |

### Stereo (SD 480×640)

| Op | Time |
|---|---|
| `StereoBM` | 3.79 ms |
| `StereoSGBM` | 21.4 ms |
| `StereoRectify` | 8 µs |
| `ReprojectTo3D` | 463 µs |

### Epipolar geometry

| Op | 20 pts | 200 pts |
|---|---|---|
| `FindFundamentalMat` | 35 µs | 26 µs |
| `FindEssentialMat` | 302 µs | 3,757 µs |
| `TriangulatePoints` | 131 µs | 1,404 µs |

| Op | Input | Time |
|---|---|---|
| `RecoverPose` | 50 pts | 144 µs |

### ArUco

| Op | Input | Time |
|---|---|---|
| `DetectAruco` raw | 400×400 scene | 275 µs |
| `DetectAruco` improc++ | 400×400 scene | 276 µs |
| `GenerateAruco` | 100×100 px | 3 µs |
| `GenerateAruco` | 200×200 px | 10 µs |
| `ArucoPose` | 1 marker | 4 µs |
| `CharucoBoard` | 5×7, 80px | 1.51 ms |

</details>

<details>
<summary><strong>Background subtraction (improc::core)</strong> — BackgroundSubtractMOG2 · BackgroundSubtractKNN</summary>

Wrapper overhead ≈ 0 for both algorithms. Times in µs/frame.

| Op | 480×640 raw | 480×640 improc | 1080×1920 raw | 1080×1920 improc |
|---|---|---|---|---|
| `BackgroundSubtractMOG2` | 287 | 288 | 1,800 | 1,800 |
| `BackgroundSubtractKNN` | 332 | 331 | 2,000 | 1,800 |

Raw JSON: [`benchmarks/results/2026-05-31-background-subtract.json`](benchmarks/results/2026-05-31-background-subtract.json)

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
| improc++ | 0.571 ms · 1,752 img/s | 0.452 ms · 2,212 img/s |
| Raw OpenCV | 0.477 ms · 2,097 img/s | 0.456 ms · 2,193 img/s |

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
| `transform \| take(16/256)` **lazy** | 591 µs | 2,200 µs |
| `transform \| take(16/256)` **eager** | 9,500 µs | 35,000 µs |
| Speedup | **16×** | **16×** |

<details>
<summary><strong>Photo + Stitching (improc::core)</strong></summary>

> Build: `cmake --build build --target improc_benchmarks`
> Run: `./build/improc_benchmarks --benchmark_filter="edge_preserving|detail_enhance|stylize|pencil|seamless|merge_hdr|tonemap|stitch"`

### Throughput — SD (480×640), Apple M4 Pro, Release

| Op | Time | Notes |
|---|---|---|
| EdgePreservingFilter | 14.7 ms | Recursive filter (default) |
| DetailEnhance | 18.5 ms | |
| Stylize | 58.0 ms | |
| PencilSketch | 14.0 ms | Returns {gray, color} |
| SeamlessClone | 10.5 ms | Normal clone mode |
| MergeHDR (Mertens) | 3.6 ms | 3 frames, 5 iterations |
| ToneMap (Reinhard) | 2.6 ms | Float32C3→BGR |
| Stitch | 2.1 ms | 2-image panorama, 5 iterations |

Raw JSON: [`benchmarks/results/2026-05-31-photo.json`](benchmarks/results/2026-05-31-photo.json)

</details>

<details>
<summary><strong>Quality + Hashing (improc::core)</strong></summary>

> Run: `./build/improc_benchmarks --benchmark_filter="psnr|ssim|gmsd|mse|hash"`

### Quality metrics — SD (480×640), Apple M4 Pro, Release

| Op | Time | Notes |
|---|---|---|
| PSNR | 458 µs | Mean over 3 channels |
| SSIM | 11.6 ms | Gaussian window (11×11, σ=1.5) |
| GMSD | 2.3 ms | Prewitt gradient, grayscale |
| MSE | 461 µs | Mean over 3 channels |

### Perceptual hashing — SD (480×640)

| Op | Time | Hash size | Notes |
|---|---|---|---|
| AverageHash | 110 µs | 8 B (64 bits) | Hamming distance |
| PHash | 106 µs | 8 B (64 bits) | DCT-based, Hamming |
| MarrHildrethHash | 985 µs | 72 B (576 bits) | LoG zero-crossing, Hamming |
| RadialVarianceHash | 1.1 ms | 40 × f64 | L2 distance |
| ColorMomentHash | 175 µs | 42 × f64 | L2 distance |
| BlockMeanHash | 854 µs | 32 B (256 bits) | 16×16 blocks, Hamming |

### Distance overhead (480×640 pre-hashed)

| | Time |
|---|---|
| `PHash::distance` (Hamming) | 108 ns |
| `RadialVarianceHash::distance` (L2) | 36 ns |

Raw JSON: [`benchmarks/results/2026-05-31-quality.json`](benchmarks/results/2026-05-31-quality.json)

</details>
