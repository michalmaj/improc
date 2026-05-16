# Namespace Structure in improc++

The library is organized into modular namespaces under the root `improc` namespace. Each namespace has a single responsibility and maps to a corresponding directory under `include/improc/` and `src/`.

## Table of Contents

- [`improc`](#improc--root-namespace-exceptions-and-error-values) — Root namespace: exceptions and error values
- [`improc::core`](#improccore--type-safe-image-primitives) — Type-safe image primitives
- [`improc::io`](#improcio--inputoutput) — Input/Output
- [`improc::ml`](#improcml--machine-learning-utilities) — Machine Learning utilities
  - [Augmentation](#augmentation-augmentationhpp)
  - [Geometric Extras (v0.4.0)](#geometric-extras-v040)
  - [Colour Extras (v0.4.0)](#colour-extras-v040)
  - [Erase / Dropout (v0.4.0)](#erase--dropout-v040)
  - [Bbox-aware ops](#bbox-aware-geometric-ops-annotatedhpp-augmentbbox_composehpp)
  - [Blur Extras (v0.4.0)](#blur-extras-v040)
  - [MixUp / CutMix (v0.4.0-C)](#mixup--cutmix-v040-c)
  - [Multi-Object Tracking (v0.5.0)](#multi-object-tracking-v050)
- [`improc::threading`](#improcthreading--concurrency-utilities) — Concurrency utilities
- [`improc::visualization`](#improcvisualization--chart-and-display-utilities) — Chart and display utilities
- [`improc::onnx`](#improconnx--onnx-runtime-inference) — ONNX Runtime inference
- [`improc::views`](#improcviews--lazy-image-pipeline) — Lazy Image Pipeline
- [Planned namespaces](#planned-namespaces)

---

## `improc` — Root namespace: exceptions and error values

**Headers:** `improc/exceptions.hpp`, `improc/error.hpp`

### Exception hierarchy (`exceptions.hpp`)

All library exceptions inherit from `improc::Exception`, which inherits from `std::exception`. Catch at the granularity you need.

```
std::exception
└── improc::Exception
    ├── improc::FormatError         — wrong image format (has expected_format / actual_format / context)
    ├── improc::ParameterError      — invalid parameter value (has param_name / constraint / context)
    ├── improc::IoError             — I/O failure
    │   ├── improc::FileNotFoundError  — path does not exist or is not a directory (has path())
    │   └── improc::CameraError        — camera open/read failure (has device_id())
    ├── improc::ModelError          — model load failure (has model_path() / reason())
    ├── improc::DataError           — dataset/loader data problem
    └── improc::AugmentError        — augmentation precondition failure
```

**Rule of thumb:** throw for programming errors (wrong format, invalid parameters, misuse of API); return `std::expected<T, Error>` for environmental errors (file not found, camera unavailable, no images loaded).

### `improc::Error` value type (`error.hpp`)

Used as the error channel in `std::expected<T, improc::Error>` returns throughout the library.

```cpp
struct Error {
    enum class Code {
        NoImages, EmptyDataset, DirectoryNotFound,
        InvalidModelFile, CameraUnavailable, CameraFrameEmpty,
        InsufficientPoints, HomographyFailed,
        ImageReadFailed,      // cv::imread returned empty mat
        ImageWriteFailed,     // cv::imwrite returned false
        OnnxModelLoadFailed,  // ORT failed to parse / load the .onnx file
        OnnxInferenceFailed,  // ORT session Run() returned an error
        OnnxSessionNotLoaded  // run() called before load()
    };
    Code        code;
    std::string message;

    // Named constructors
    static Error no_images(const std::string& dir);
    static Error directory_not_found(const std::string& path);
    static Error invalid_model_file(const std::string& path, const std::string& reason);
    static Error camera_unavailable(int device_id);
    static Error camera_frame_empty(int device_id);
    static Error insufficient_points(std::size_t got);
    static Error homography_failed();
    static Error image_read_failed(const std::string& path);
    static Error image_write_failed(const std::string& path);
    static Error onnx_model_load_failed(const std::string& path, const std::string& reason);
    static Error onnx_inference_failed(const std::string& reason);
    static Error onnx_session_not_loaded();
};
```

Access the human-readable description via `.message`:

```cpp
auto result = loader.get_images();
if (!result)
    std::cerr << result.error().message << '\n';
```

---

## `improc::core` — Type-safe image primitives

**Status: Implemented**

Provides a compile-time type-safe wrapper over `cv::Mat`, format conversion utilities, and a pipeline composition operator.

### Format tag types (`format_traits.hpp`)

Empty structs used as template parameters. `FormatTraits<T>` maps each tag to its OpenCV type constant and channel count.

| Tag | `cv_type` | `channels` | `is_float` |
|---|---|---|---|
| `BGR` | `CV_8UC3` | 3 | `false` |
| `Gray` | `CV_8UC1` | 1 | `false` |
| `BGRA` | `CV_8UC4` | 4 | `false` |
| `Float32` | `CV_32FC1` | 1 | `true` |
| `Float32C3` | `CV_32FC3` | 3 | `true` |
| `HSV` | `CV_8UC3` | 3 | `false` |
| `LAB` | `CV_8UC3` | 3 | `false` |
| `YCrCb` | `CV_8UC3` | 3 | `false` |

**`struct HSV {}`** — format tag for HSV color space (H∈[0,179], S∈[0,255], V∈[0,255]).

**`struct LAB {}`** — format tag for CIE L\*a\*b\* color space. 8-bit encoding: L ∈ [0, 255] (L* scaled by 255/100), a ∈ [0, 255] (a* shifted by +128), b ∈ [0, 255] (b* shifted by +128).

**`struct YCrCb {}`** — format tag for YCbCr color space (OpenCV channel order: Y, Cr, Cb). 8-bit, 3 channels. Y carries luminance; Cr and Cb carry colour difference.

All `FormatTraits<F>` specializations include `static constexpr bool is_float` — `false` for BGR/Gray/BGRA/HSV, `true` for Float32/Float32C3.

### `Image<Format>` (`image.hpp`)

Thin wrapper over `cv::Mat` with shallow-copy semantics (same as `cv::Mat`). Constructor validates that the `cv::Mat` is non-empty and its type matches `FormatTraits<Format>::cv_type` — throws `FormatError` on type mismatch, `ParameterError` on empty mat.

```cpp
Image<BGR> img(mat);            // validated at construction
Image<BGR> copy = img;          // shallow copy (shared data)
Image<BGR> owned = img.clone(); // deep copy
cv::Mat& raw = img.mat();       // explicit interop with OpenCV
```

### `convert<To, From>()` (`convert.hpp`)

Free function template. Primary template is `= delete` — unsupported conversions are compile errors. Explicit specializations define allowed conversions:

- `BGR` ↔ `Gray`
- `BGR` ↔ `BGRA`
- `Gray` ↔ `Float32` (scale ×1/255 / ×255)
- `BGR` ↔ `Float32C3` (scale ×1/255 / ×255, 3-channel float)
- `BGR` ↔ `HSV` (OpenCV `COLOR_BGR2HSV` / `COLOR_HSV2BGR`)
- `BGR` ↔ `LAB` (OpenCV `COLOR_BGR2Lab` / `COLOR_Lab2BGR`)
- `BGR` ↔ `YCrCb` (OpenCV `COLOR_BGR2YCrCb` / `COLOR_YCrCb2BGR`)

```cpp
Image<Gray>     gray  = convert<Gray, BGR>(bgr_image);
Image<Float32C3> f32c3 = convert<Float32C3, BGR>(bgr_image);
```

### Pipeline `operator|` and ops (`pipeline.hpp`)

Composes processing steps using the pipe operator. Each step is a small functor. `pipeline.hpp` is the single umbrella include — it pulls in all ops.

```cpp
// Conversion functors
Image<Float32>   result = bgr_image | ToGray{} | ToFloat32{};
Image<Float32C3> result = bgr_image | ToFloat32C3{};
Image<HSV>       hsv    = bgr_image | ToHSV{};
Image<LAB>       lab    = bgr_image | ToLAB{};
Image<YCrCb>     ycrcb  = bgr_image | ToYCrCb{};
Image<BGR>       bgr    = hsv_image | ToBGR{};  // also accepts Image<Gray>, Image<LAB>, Image<YCrCb>

// Geometric ops — templated, work on any Image<Format>
Image<BGR> r = src | Resize{}.width(224).height(224);   // both dims
Image<BGR> r = src | Resize{}.width(224);               // aspect-ratio preserved
Image<BGR> c = src | Crop{}.x(10).y(10).width(100).height(100);
Image<BGR> f = src | Flip{Axis::Horizontal};            // Axis: Horizontal/Vertical/Both
Image<BGR> t = src | Rotate{}.angle(30.0).scale(0.5);  // scale optional, default 1.0

// Normalization ops — Image<Float32> only
Image<Float32> n  = f32 | Normalize{};                  // auto min-max → [0, 1]
Image<Float32> n2 = f32 | NormalizeTo{-1.0f, 1.0f};    // explicit target range
Image<Float32> n3 = f32 | Standardize{0.485f, 0.229f}; // z-score

// Blur ops — templated, work on any Image<Format>
Image<BGR>  blurred = src  | GaussianBlur{}.kernel_size(5).sigma(1.0);
Image<Gray> median  = gray | MedianBlur{}.kernel_size(3);

// Morphological ops — templated, work on any Image<Format>
Image<Gray> dilated = mask | Dilate{}.kernel_size(5).shape(MorphShape::Ellipse);
Image<Gray> eroded  = mask | Erode{}.kernel_size(3).iterations(2);

// MorphOpen (erode→dilate): removes small bright noise
// MorphClose (dilate→erode): fills small dark holes
Image<Gray> opened = noisy | MorphOpen{}.kernel_size(3);
Image<Gray> closed = holed | MorphClose{}.kernel_size(3);

// MorphGradient — dilate − erode; highlights object boundaries
// TopHat — src − MorphOpen; reveals bright features smaller than kernel
// BlackHat — MorphClose − src; reveals dark features smaller than kernel
Image<Gray> grad     = mask | MorphGradient{}.kernel_size(3);
Image<Gray> tophat   = mask | TopHat{}.kernel_size(5);
Image<Gray> blackhat = mask | BlackHat{}.kernel_size(5);

// Threshold — templated; Otsu only valid on Image<Gray> (CV_8U)
Image<Gray> binary = gray | Threshold{}.value(128).mode(ThresholdMode::Binary);
Image<Gray> otsu   = gray | Threshold{}.mode(ThresholdMode::Otsu);

// AdaptiveThreshold — Image<Gray> only; threshold computed locally per pixel
Image<Gray> adapt  = gray | AdaptiveThreshold{}.block_size(11).C(2);
Image<Gray> adapt2 = gray | AdaptiveThreshold{}.method(AdaptiveMethod::Mean).block_size(31).C(5);
Image<Gray> adapt3 = gray | AdaptiveThreshold{}.block_size(11).C(2).invert();

// Invert — any format; bitwise NOT, applying twice restores the original
Image<Gray> inv_gray = gray | Invert{};
Image<BGR>  inv_bgr  = bgr  | Invert{};

// InRange — any input format, always returns Image<Gray>
// Output pixel is 255 when all channels are within [lower, upper]
Image<Gray> mask      = gray | InRange{}.lower({100}).upper({200});
Image<Gray> green_msk = bgr  | InRange{}.lower({0, 200, 0}).upper({50, 255, 50});

// Padding ops — templated, work on any Image<Format>
Image<BGR>  padded = src | Pad{}.top(10).bottom(10).left(20).right(20).mode(PadMode::Reflect);
Image<BGR>  square = src | PadToSquare{}.value({114, 114, 114});  // letterbox for inference
Image<BGR> cc  = src | CenterCrop{}.width(224).height(224);           // centered crop
Image<BGR> lb  = src | LetterBox{}.width(640).height(640);            // resize+pad to canvas
Image<BGR> lb2 = src | LetterBox{}.width(640).height(640).value({0, 0, 0}); // black fill

// CLAHE — Image<Gray> (direct) or Image<BGR> (applied to L channel in LAB)
Image<Gray> eq  = gray | CLAHE{};                                  // defaults: clip=40, tile=8×8
Image<Gray> eq2 = gray | CLAHE{}.clip_limit(2.0).tile_grid_size(8, 8);
Image<BGR>  eq3 = bgr  | CLAHE{}.clip_limit(3.0);                 // colour-safe via LAB
```

**`AdaptiveThreshold`** accepts only `Image<Gray>`. Throws `ParameterError` if `block_size` is even or < 3. Default: Gaussian method, Binary output, block_size = 11, C = 2.0.

**`Invert`** works on integer formats (Gray, BGR, BGRA). Applies `cv::bitwise_not`; for 8-bit images, channel value v becomes 255 − v. Float formats are not supported. Applying twice returns the original image.

**`InRange`** accepts any input format and always returns `Image<Gray>`. Throws `ParameterError` if either `lower` or `upper` is not set. Both bounds are inclusive.

**`MorphOpen`** and **`MorphClose`** share the same parameter contract as `Dilate`/`Erode`: `kernel_size` must be odd and positive; `iterations` must be >= 1. `MorphOpen` removes isolated bright regions (noise); `MorphClose` fills isolated dark regions (holes).

**`MorphGradient`**, **`TopHat`**, and **`BlackHat`** throw `ParameterError` if `kernel_size` is not odd and positive. They have no `iterations` setter. `MorphGradient` computes dilate − erode (boundary highlight); `TopHat` computes src − MorphOpen (small bright features on dark background); `BlackHat` computes MorphClose − src (small dark features on bright background).

**Geometric ops** throw `ParameterError` when required parameters are missing or invalid (e.g. no dimension in `Resize`, ROI out of bounds in `Crop`, no angle in `Rotate`).

**`CenterCrop`** throws `ParameterError` if either dimension is missing, non-positive, or exceeds the source image size.

**`LetterBox`** throws `ParameterError` if either dimension is missing or non-positive. Output is always exactly `width × height` pixels; default fill color is `{114, 114, 114}` (YOLO convention).

**Normalization ops** throw `ParameterError` at construction if parameters are invalid (`NormalizeTo` requires `min < max`; `Standardize` requires `std_dev > 0`). A uniform image passed to `Normalize` or `NormalizeTo` returns a zero-filled image.

**`CLAHE`** throws `ParameterError` for non-positive `clip_limit` or tile dimensions. On BGR input the operation is colour-safe: it converts to LAB, equalises the L channel only, and converts back — so hue and saturation are preserved.

```cpp
// GammaCorrection — any format (Gray, BGR, Float32, Float32C3)
// output = input ^ gamma  (gamma > 1 darkens, gamma < 1 brightens)
Image<BGR>  bright = img | GammaCorrection{}.gamma(0.5f);
Image<Gray> dark   = gray | GammaCorrection{}.gamma(2.0f);
Image<Float32C3> fp = fp_img | GammaCorrection{}.gamma(1.8f);

// BilateralFilter — edge-preserving smoothing (Gray or BGR, 8-bit)
Image<BGR>  smooth = img | BilateralFilter{}.diameter(9).sigma_color(75).sigma_space(75);
Image<Gray> s2     = gray | BilateralFilter{};  // defaults: d=9, sigma_color=75, sigma_space=75

// HistogramEqualization — global contrast normalization; no parameters
Image<Gray> heq_gray = gray | HistogramEqualization{};
Image<BGR>  heq_bgr  = bgr  | HistogramEqualization{};  // equalizes Y channel in YCrCb

// NLMeansDenoising — Non-Local Means noise reduction
Image<Gray> nlm_gray = noisy | NLMeansDenoising{}.h(10.0f);
Image<BGR>  nlm_bgr  = noisy | NLMeansDenoising{}.h(10.0f).h_color(10.0f);
// defaults: h=3.0, h_color=3.0, template_window_size=7, search_window_size=21

// SobelEdge — gradient magnitude; accepts Gray or BGR (auto-converts to Gray)
// ksize must be 1, 3, 5, or 7
Image<Gray> sob  = gray | SobelEdge{}.ksize(3);
Image<Gray> sob2 = bgr  | SobelEdge{};  // BGR auto-converted to Gray before processing

// CannyEdge — hysteresis thresholding; accepts Gray or BGR
// aperture_size must be 3, 5, or 7
Image<Gray> canny = gray | CannyEdge{}.threshold1(50).threshold2(150);
Image<Gray> c2    = bgr  | CannyEdge{}.threshold1(100).threshold2(200).aperture_size(3);

// LaplacianEdge — second-derivative edge detection; accepts Gray or BGR
// ksize must be odd and positive; scale must be > 0
Image<Gray> lap  = gray | LaplacianEdge{};
Image<Gray> lap2 = bgr  | LaplacianEdge{}.ksize(3).scale(2.0).delta(128.0);

// HarrisCorner — corner response map normalized to [0, 255]; accepts Gray or BGR
// ksize must be 3, 5, or 7; k must be in (0, 1); block_size must be > 0
Image<Gray> corners = gray | HarrisCorner{};
Image<Gray> c2      = bgr  | HarrisCorner{}.block_size(3).ksize(5).k(0.05);

// PyrDown — halves image dimensions (Gaussian pyramid down-step)
// PyrUp   — doubles image dimensions (Gaussian pyramid up-step)
// Both work on any Image<Format>; no parameters.
Image<Gray> small  = img  | PyrDown{};
Image<BGR>  large  = bgr  | PyrUp{};
Image<Gray> down2  = img  | PyrDown{} | PyrDown{};  // two levels down

// Drawing ops — all accept and return Image<BGR>; source is never mutated
Image<BGR> labeled   = bgr | DrawText{"Score: 0.95"}.position({10, 30}).color({0, 255, 0});
Image<BGR> with_line = bgr | DrawLine{{0, 0}, {320, 240}}.color({255, 0, 0}).thickness(2);
Image<BGR> with_circ = bgr | DrawCircle{{160, 120}, 50}.color({0, 0, 255});
Image<BGR> filled_c  = bgr | DrawCircle{{160, 120}, 50}.thickness(-1);  // filled
Image<BGR> with_rect = bgr | DrawRectangle{cv::Rect{10, 10, 100, 80}}.color({255, 255, 0});
Image<BGR> filled_r  = bgr | DrawRectangle{cv::Rect{10, 10, 100, 80}}.thickness(-1);

// FindContours — extracts contours from a binary Image<Gray>; returns ContourSet (not Image)
// DrawContours — draws ContourSet onto a BGR clone; index(-1) = all, thickness(-1) = fill
ContourSet cs      = binary | FindContours{};
ContourSet cs_tree = binary | FindContours{}.mode(FindContours::Mode::Tree);
Image<BGR> drawn   = bgr | DrawContours{cs}.color({0, 255, 0});
Image<BGR> filled  = bgr | DrawContours{cs}.thickness(-1);
Image<BGR> first   = bgr | DrawContours{cs}.index(0).color({255, 0, 0});

// ContourSet accessors
std::size_t n    = cs.size();
double      a    = cs.area(0);        // cv::contourArea
double      p    = cs.perimeter(0);   // cv::arcLength (closed)
cv::Rect    br   = cs.bounding_rect(0);

// ConnectedComponents — labels connected regions in a binary Image<Gray>; returns ComponentMap (not Image)
// ComponentMap — label 0 = background; labels 1..N-1 = components
ComponentMap cm   = binary | ConnectedComponents{};
ComponentMap cm4  = binary | ConnectedComponents{}.connectivity(ConnectedComponents::Connectivity::Four);
int          n_labels = cm.count();      // total label count (including background)
int          comp_area = cm.area(1);    // pixel area of label 1
cv::Rect     br   = cm.bounding_rect(1);
cv::Point2d  cen  = cm.centroid(1);
cv::Mat      mask = cm.mask(1);          // CV_8U mask: 255 where label==1

// DistanceTransform — Euclidean (or other) distance to nearest 0-pixel; returns Image<Float32>
Image<Float32> dt    = binary | DistanceTransform{};
Image<Float32> dt_l1 = binary | DistanceTransform{}.dist_type(DistanceTransform::DistType::L1);
Image<Float32> dt_m5 = binary | DistanceTransform{}.mask_size(DistanceTransform::MaskSize::Mask5);

// DetectORB / DetectSIFT / DetectAKAZE — keypoint detection; all return KeypointSet (not Image)
// KeypointSet — plain struct: public `keypoints` vector + size()/empty() helpers
KeypointSet ks_orb   = gray | DetectORB{};
KeypointSet ks_sift  = gray | DetectSIFT{};
KeypointSet ks_akaze = gray | DetectAKAZE{};

KeypointSet few_orb  = gray | DetectORB{}.max_features(50);
KeypointSet few_sift = gray | DetectSIFT{}.max_features(50);
KeypointSet strict   = gray | DetectAKAZE{}.threshold(0.01f);

std::size_t n_kp     = ks_orb.size();
if (!ks_orb.empty())
    cv::KeyPoint kp  = ks_orb.keypoints[0];   // direct vector access

// DescribeORB / DescribeSIFT / DescribeAKAZE — compute descriptors from keypoints
// DescriptorSet — owns KeypointSet + cv::Mat descriptors (CV_32F for SIFT; CV_8U for ORB/AKAZE)
DescriptorSet desc_orb   = gray | DescribeORB{ks_orb};
DescriptorSet desc_sift  = gray | DescribeSIFT{ks_sift};
DescriptorSet desc_akaze = gray | DescribeAKAZE{ks_akaze};

// Works on Image<BGR> too — auto-converts to Gray internally
Image<BGR> bgr_img{cv::Mat(200, 200, CV_8UC3, cv::Scalar(128, 128, 128))};
DescriptorSet desc_bgr = bgr_img | DescribeORB{ks_orb};

std::size_t n_desc = desc_orb.size();          // == desc_orb.keypoints.size()
int desc_rows = desc_orb.descriptors.rows;     // == n_desc
int desc_type = desc_orb.descriptors.type();   // CV_8U for ORB/AKAZE, CV_32F for SIFT

// MatchBF / MatchFlann — callable structs: Op{desc1, desc2}() — NOT pipeline ops
// MatchSet — plain struct: public `matches` vector + size()/empty() helpers
MatchSet ms_bf    = MatchBF{desc_orb, desc_orb}();                           // brute-force, auto norm
MatchSet ms_cross = MatchBF{desc_orb, desc_orb}.cross_check(true)();         // mutual NN filter
MatchSet ms_dist  = MatchBF{desc_orb, desc_orb}.max_distance(50.0f)();       // distance filter
MatchSet ms_sift  = MatchBF{desc_sift, desc_sift}();                         // SIFT: auto L2 norm

MatchSet ms_flann = MatchFlann{desc_sift, desc_sift}();                       // Lowe ratio 0.7
MatchSet ms_tight = MatchFlann{desc_sift, desc_sift}.ratio_threshold(0.5f)(); // stricter

std::size_t n_matches = ms_bf.size();
for (const cv::DMatch& m : ms_bf.matches)
    (void)m.distance;  // Hamming for ORB/AKAZE, L2 for SIFT

// DrawKeypoints — pipeline op: draws keypoints; returns Image<BGR> from Gray or BGR input
// DrawMatches   — callable: side-by-side match visualisation; NOT a pipeline op
Image<BGR> kp_vis   = gray   | DrawKeypoints{ks_orb};           // Gray → BGR with keypoints
Image<BGR> kp_vis2  = bgr_img | DrawKeypoints{ks_orb};          // BGR input also accepted

// DrawMatches takes both images, both KeypointSets, and the MatchSet
Image<BGR> match_vis = DrawMatches{bgr_img, ks_orb, bgr_img, ks_orb, ms_bf}();
// output width = img1.cols + img2.cols; height = max(img1.rows, img2.rows)

// WarpPerspective — apply a 3×3 homography to an image
Image<BGR> warped = src | WarpPerspective{}.homography(H).width(640).height(480);
```

**`GammaCorrection`** throws `ParameterError` if `gamma <= 0`. Uses an 8-bit LUT for integer formats (fast); `cv::pow` + clamp for float formats.

**`BilateralFilter`** throws `ParameterError` for non-positive `diameter`, `sigma_color`, or `sigma_space`. Supports `Gray` and `BGR` only (OpenCV bilateral requires 8-bit input).

**`HistogramEqualization`** has no parameters. On `Image<Gray>` it calls `cv::equalizeHist` directly; on `Image<BGR>` it converts to YCrCb, equalizes the Y (luminance) channel only, and converts back — preserving hue and saturation.

**`NLMeansDenoising`** throws `ParameterError` if `h` or `h_color` are not positive, or if `template_window_size` / `search_window_size` are not odd and positive. `h_color` is silently ignored when applied to `Image<Gray>`. Larger `h` removes more noise but may blur fine details.

**`SobelEdge`** throws `ParameterError` if `ksize` is not in {1, 3, 5, 7}. Computes X and Y gradients in CV_32F, combines via `cv::magnitude`, and converts back to CV_8U.

**`CannyEdge`** throws `ParameterError` for negative thresholds or invalid `aperture_size` (not in {3, 5, 7}).

**`LaplacianEdge`** throws `ParameterError` if `ksize` is not odd and positive, or if `scale` is not positive. `delta` accepts any value. Defaults: `ksize=1`, `scale=1.0`, `delta=0.0`. Uses CV_16S intermediate depth to preserve negative responses, then `cv::convertScaleAbs` to fold into CV_8U.

**`HarrisCorner`** throws `ParameterError` if `block_size <= 0`, if `ksize` is not in {3, 5, 7}, or if `k` is not in (0, 1). Returns a corner response map normalized to [0, 255] — brighter pixels indicate stronger corner response. Defaults: `block_size=2`, `ksize=3`, `k=0.04`.

**`PyrDown`** reduces image dimensions to `ceil(rows/2) × ceil(cols/2)` using Gaussian pyramid blurring (`cv::pyrDown`). Works on any `Image<Format>`. No parameters; no error conditions.

**`PyrUp`** doubles image dimensions to `2*rows × 2*cols` using Gaussian pyramid upsampling (`cv::pyrUp`). Works on any `Image<Format>`. No parameters; no error conditions.

**`DrawText`** renders text on a clone of the BGR image using `cv::putText` (font: `FONT_HERSHEY_SIMPLEX`, antialiased). Defaults: `position=(10,30)`, `font_scale=1.0`, `color=(0,255,0)`, `thickness=1`. Throws `ParameterError` if `font_scale <= 0` or `thickness <= 0`.

**`DrawLine`** draws an antialiased line (`cv::line`, `LINE_AA`) on a clone. Defaults: `color=(0,255,0)`, `thickness=1`. Throws `ParameterError` if `thickness <= 0`.

**`DrawCircle`** draws an antialiased circle (`cv::circle`, `LINE_AA`) on a clone. Throws `ParameterError` at construction if `radius <= 0`. `thickness(-1)` fills the circle. Throws `ParameterError` for thickness not in {−1} ∪ ℤ⁺. Defaults: `color=(0,255,0)`, `thickness=1`.

**`DrawRectangle`** draws an antialiased rectangle (`cv::rectangle`, `LINE_AA`) on a clone. `thickness(-1)` fills the rectangle. Throws `ParameterError` for thickness not in {−1} ∪ ℤ⁺. Defaults: `color=(0,255,0)`, `thickness=1`.

**`FindContours`** finds contours in a binary `Image<Gray>` using `cv::findContours`. Returns `ContourSet` — not `Image<>` — so the pipe `gray | FindContours{}` short-circuits the normal image-to-image pipeline. Mode defaults to `External` (outermost contours only); method defaults to `Simple` (compressed horizontal/vertical/diagonal segments). No parameters throw; all combinations are valid.

**`ContourSet`** — result type with public members `contours` (`std::vector<std::vector<cv::Point>>`) and `hierarchy` (`std::vector<cv::Vec4i>`). Convenience methods: `size()`, `empty()`, `area(i)` (`cv::contourArea`), `perimeter(i)` (`cv::arcLength`, closed), `bounding_rect(i)` (`cv::boundingRect`). Index-checked (`.at()`): out-of-bounds throws `std::out_of_range`.

**`DrawContours`** draws contours from a `ContourSet` onto a BGR image clone using `cv::drawContours`. `index(-1)` draws all (default); pass a non-negative index to draw a single contour. `thickness(-1)` fills contours. Throws `ParameterError` for thickness not in {−1} ∪ ℤ⁺. Defaults: `index=-1`, `color=(0,255,0)`, `thickness=1`.

**`ConnectedComponents`** labels connected regions in a binary `Image<Gray>` using `cv::connectedComponentsWithStats`. Returns `ComponentMap` — not `Image<>`. Label 0 is always the background. Connectivity defaults to `Eight`; use `Four` for stricter adjacency. No parameters throw.

**`ComponentMap`** — result type with public members `labels` (CV_32S, same size as source), `stats` (N×5 int matrix), `centroids` (N×2 double matrix), and `num_labels` (int, including background). `count()` is an alias for `num_labels`. Accessors `area(i)`, `bounding_rect(i)`, `centroid(i)`, `mask(i)` are bounds-checked and throw `std::out_of_range` for invalid label (including -1 or label ≥ count()).

**`DistanceTransform`** computes the distance of each non-zero pixel to the nearest zero pixel in a binary `Image<Gray>` using `cv::distanceTransform`. Returns `Image<Float32>`. Distance type defaults to `L2` (Euclidean); `L1` and `C` (Chebyshev) available. Mask size defaults to `Mask3`; `Mask5` and `Precise` also available. No parameters throw.

**`KeypointSet`** — result type with a public `keypoints` (`std::vector<cv::KeyPoint>`) member plus `size()` and `empty()` helpers. No bounds checking — use standard vector access. A default-constructed `KeypointSet` is empty.

**`DetectORB`** detects ORB keypoints in `Image<Gray>` using `cv::ORB`. Returns `KeypointSet`. Fluent setters: `max_features(n)` (default 500), `scale_factor(f)` (default 1.2), `n_levels(n)` (default 8). `max_features` strictly limits the returned count. No parameters throw.

**`DetectSIFT`** detects SIFT keypoints in `Image<Gray>` using `cv::SIFT` (main OpenCV ≥ 4.4). Returns `KeypointSet`. Fluent setters: `max_features(n)` (default 0 = no limit), `n_octave_layers(n)` (default 3). No parameters throw.

**`DetectAKAZE`** detects AKAZE keypoints in `Image<Gray>` using `cv::AKAZE`. Returns `KeypointSet`. Fluent setter: `threshold(f)` (default 0.001f). Higher threshold → stricter → fewer keypoints. No parameters throw.

**`DescriptorSet`** — result type with a `KeypointSet keypoints` member and a `cv::Mat descriptors` member. `size()` returns `keypoints.size()`, which equals `descriptors.rows` after computation; `empty()` returns `true` when the stored `KeypointSet` is empty. Descriptor type: CV_32F for SIFT; CV_8U for ORB and AKAZE. A default-constructed `DescriptorSet` is empty. The underlying OpenCV `cv::Feature2D::compute()` call may prune keypoints that cannot be described; the stored `keypoints` reflects the pruned set.

**`DescribeORB`** computes ORB descriptors (CV_8U, 32 bytes per keypoint) for a `KeypointSet`. Constructed with an explicit `KeypointSet`: `DescribeORB{kps}`. Accepts `Image<Gray>` or `Image<BGR>` (auto-converts to Gray). An empty input `KeypointSet` yields an empty `DescriptorSet`. No parameters throw.

**`DescribeSIFT`** computes SIFT descriptors (CV_32F, 128 floats per keypoint) for a `KeypointSet`. Constructed with an explicit `KeypointSet`: `DescribeSIFT{kps}`. Accepts `Image<Gray>` or `Image<BGR>`. No parameters throw.

**`DescribeAKAZE`** computes AKAZE descriptors (CV_8U) for a `KeypointSet`. Constructed with an explicit `KeypointSet`: `DescribeAKAZE{kps}`. Accepts `Image<Gray>` or `Image<BGR>`. No parameters throw.

**`MatchSet`** — result type with a public `matches` (`std::vector<cv::DMatch>`) member plus `size()` and `empty()` helpers. A default-constructed `MatchSet` is empty.

**`MatchBF`** performs brute-force descriptor matching. Constructed with two `DescriptorSet` objects: `MatchBF{desc1, desc2}`. Norm type is auto-detected: `NORM_HAMMING` for CV_8U (ORB/AKAZE), `NORM_L2` for CV_32F (SIFT). Returns empty `MatchSet` for empty input. Fluent setters: `cross_check(bool)` (default false), `max_distance(f)` (default 0 = no filter; positive value keeps only matches with distance ≤ f). Invoked as `MatchBF{desc1, desc2}()`. Throws `ParameterError` if `max_distance < 0`.

**`MatchFlann`** performs FLANN-based matching with Lowe ratio test. Constructed with two `DescriptorSet` objects: `MatchFlann{desc1, desc2}`. Runs `knnMatch(k=2)` and keeps matches where `distance1 < ratio_threshold × distance2`. Accepts CV_32F (float) descriptors only — throws `ParameterError` at call time for binary (CV_8U) descriptors. Returns empty `MatchSet` for empty input. Fluent setter: `ratio_threshold(f)` (default 0.7f; must be in (0, 1]; Lowe's recommended value is 0.7–0.8). Throws `ParameterError` if `ratio_threshold` is out of range (at setter) or if descriptors are binary (at call time).

**`DrawKeypoints`** is a pipeline op that draws keypoints onto an image using `cv::DRAW_RICH_KEYPOINTS` (oriented, scaled circles). Constructed with an explicit `KeypointSet`: `DrawKeypoints{kps}`. Accepts `Image<Gray>` or `Image<BGR>`; always returns `Image<BGR>`. An empty `KeypointSet` produces a valid BGR image with no annotations. No parameters throw.

**`DrawMatches`** renders two `Image<BGR>` images side-by-side with connecting lines for each match. Constructed with `DrawMatches{img1, kps1, img2, kps2, matches}` and invoked as `DrawMatches{...}()`. Output width is `img1.cols + img2.cols`; height is `max(img1.rows, img2.rows)`. An empty `MatchSet` produces a valid side-by-side image with no lines. No parameters throw.

**`ToLAB`** converts BGR → CIE L\*a\*b\* using `cv::COLOR_BGR2Lab`. Output is `Image<LAB>` (CV_8UC3). OpenCV 8-bit encoding: L ∈ [0, 255], a ∈ [0, 255], b ∈ [0, 255] (L\* scaled by 255/100; a\* and b\* shifted by +128). Round-trip BGR → LAB → BGR introduces at most 2 per channel from quantization. No parameters; no error conditions.

**`ToYCrCb`** converts BGR → YCrCb using `cv::COLOR_BGR2YCrCb`. Output is `Image<YCrCb>` (CV_8UC3). The Y channel carries luminance; Cr and Cb carry colour difference. Useful for luminance-only ops (e.g. histogram equalization on Y only). Round-trip BGR → YCrCb → BGR introduces at most 2 per channel from quantization. No parameters; no error conditions.

**`ToBGR`** accepts `Image<Gray>`, `Image<HSV>`, `Image<LAB>`, and `Image<YCrCb>` — all return `Image<BGR>`. No error conditions.

**`find_homography(src, dst, threshold)`** — free function; computes a 3×3 homography matrix from ≥4 corresponding point pairs via RANSAC. Returns `std::expected<cv::Mat, Error>` — error if fewer than 4 points are provided or RANSAC fails to find a valid homography.

| Function / Op | Return type | Description |
|---|---|---|
| `find_homography(src, dst, threshold)` | `std::expected<cv::Mat, Error>` | Compute 3×3 homography from ≥4 point pairs via RANSAC |
| `WarpPerspective` | pipeline op | Apply homography; `.homography(H)`, `.width(w)`, `.height(h)` |
| `WarpAffine` | pipeline op | Apply a 2×3 affine transformation; `.matrix(M)`, `.width(w)`, `.height(h)` |
| `ApplyMask` | pipeline op | Zero pixels outside a binary `Image<Gray>` mask; `.mask(m)` |
| `UnsharpMask` | pipeline op | Sharpen via blur subtraction; `.sigma(s)`, `.strength(f)` |

**`WarpPerspective`** throws `ParameterError` if `.homography()` is not set, or if width/height are not positive.

```cpp
// Brightness and Contrast — work on any Image<Format>
Image<BGR>  bright   = img  | Brightness{}.delta(30.0);   // additive; clamped to valid range
Image<Gray> dark     = gray | Brightness{}.delta(-20.0);
Image<BGR>  contrast = img  | Contrast{}.factor(1.5);     // multiplicative

// WeightedBlend — blends two same-format, same-size images
Image<BGR>  blended  = img  | WeightedBlend<BGR>{img2}.alpha(0.6);  // out = α·img + (1−α)·img2

// AlphaBlend — composites a BGRA overlay onto a BGR background
Image<BGR>  composited = bg | AlphaBlend{overlay_bgra};   // per-pixel alpha from overlay's A channel
```

**`Brightness`** — additive brightness adjustment via `.delta(double)`. Works on any `AnyFormat`. Pixel values are clamped to the valid range for the format.

**`Contrast`** — multiplicative contrast scaling via `.factor(double)`. Works on any `AnyFormat`. Throws `ParameterError` if `factor <= 0`.

**`WeightedBlend<F>`** — weighted average of two same-format images. `.alpha(double)` controls the blend weight of the first image (second image weight = 1−alpha). Throws `ParameterError` if `alpha` is outside `[0, 1]` or if image sizes differ.

**`AlphaBlend`** — composites a BGRA overlay image onto a BGR background using per-pixel alpha from the overlay's A channel. Throws `ParameterError` if overlay and background sizes differ.

---

## `improc::io` — Input/Output

**Status: Implemented**

### `imread<F>` / `imwrite` (`io/image_io.hpp`)

Typed image I/O functions. Both return `std::expected` for error handling — no exceptions thrown for environmental failures.

```cpp
#include "improc/io/image_io.hpp"
using namespace improc::io;

// Read and auto-convert to the requested format
auto result = imread<BGR>("/path/to/image.png");
if (!result)
    std::cerr << result.error().message << '\n';
else
    Image<BGR> img = *result;

// Read directly as Gray or Float32C3
auto gray = imread<Gray>("/path/to/image.png");
auto fp   = imread<Float32C3>("/path/to/image.png");

// Write any Image<F> to file (format inferred from extension)
auto ok = imwrite("/path/to/output.png", img);
if (!ok)
    std::cerr << ok.error().message << '\n';
```

`imread<F>` calls `cv::imread` then `convert<F, BGR>()` to produce the requested format. Returns `Error{Code::ImageReadFailed}` if `cv::imread` returns an empty mat.

`imwrite` calls `cv::imwrite`. Returns `Error{Code::ImageWriteFailed}` if `cv::imwrite` returns false (bad path, unsupported extension, permissions error).

### `CameraCapture` (`io/camera_capture.hpp`)

Asynchronous camera frame capture. Runs a background capture thread. Non-copyable, non-movable.

`getFrame()` returns `std::expected<cv::Mat, improc::Error>` — use `.has_value()` / `*result` / `.error()` to inspect the result.

```cpp
CameraCapture cam(0);
std::this_thread::sleep_for(std::chrono::milliseconds(500)); // warm-up

if (auto frame = cam.getFrame())
    cv::imshow("Live", *frame);
else
    std::cerr << frame.error().message << '\n';

cam.stop();
```

### `VideoWriter` (`io/video_writer.hpp`)

Synchronous RAII video writer. Size is auto-detected from the first frame if not set explicitly. Codec is auto-detected from the file extension. `operator()` writes a frame and returns it unchanged — pipeline-compatible.

| Extension | Auto codec |
|---|---|
| `.mp4`, `.mov`, `.m4v` | `mp4v` |
| `.avi` | `MJPG` |
| `.mkv` | `XVID` |

```cpp
// Basic usage — RAII, destructor finalises the file
{
    VideoWriter w{"output.mp4"};
    w.fps(30);
    for (auto& frame : frames)
        w(frame);
}  // file closed here

// Pipeline form — display and record at the same time
img | Show{"preview"}.wait_ms(1) | writer;

// Explicit configuration
VideoWriter w{"output.avi"};
w.fps(25).size(640, 480).codec("MJPG");
```

Throws `ParameterError` on invalid setter arguments; throws `IoError` if the underlying `cv::VideoWriter` fails to open or if a frame's size does not match the writer's configured size.

### `VideoReader` (`io/video_reader.hpp`)

Sequential video file reader. `next()` returns `std::optional<Image<BGR>>` — reads frames one by one until EOF, then returns `std::nullopt`.

---

## `improc::ml` — Machine Learning utilities

**Status: Implemented**

### `ImageLoader`

Loads all valid images (`.jpg`, `.jpeg`, `.png`) from a directory. `load_images()` throws `FileNotFoundError` if the path is not a directory. `get_images()` returns `std::expected<std::vector<cv::Mat>, improc::Error>` — error code `NoImages` if nothing was loaded.

### `Dataset`

Loads an image dataset from a directory of class-labeled subdirectories. Splits into train/val/test sets with configurable ratios and optional per-class cap.

```
root_dir/
  class_a/
    img1.jpg
  class_b/
    img2.png
```

### `ModelLoaderBase<Derived, ModelType>` (CRTP)

Base class for model loaders. Validates file path and extension (`.yml`, `.yaml`, `.xml`). Concrete loaders implement `load_impl()` and `get_impl()`.

### `HaarCascadeLoader`

Loads OpenCV Haar Cascade XML files into `cv::CascadeClassifier`.

### DNN Inference (`dnn_classifier.hpp`, `dnn_detector.hpp`, `dnn_forward.hpp`)

Standalone inference functors backed by OpenCV DNN. All accept `const Image<BGR>&` and work as both standalone functors and terminal `operator|` pipeline ops. Model is loaded at construction via `cv::dnn::readNet` — throws `ModelError` on failure (file not found, unsupported extension, parse error, empty net). Supported formats: `.onnx`, `.pb`, `.caffemodel`, `.weights`, `.t7`, `.net`.

Shared result types are in `result_types.hpp`: `ClassResult{class_id, score, label}` and `Detection{box, class_id, confidence, label}`.

```cpp
// Classification — top-K results
auto results = img | Resize{}.width(224) | DnnClassifier{"resnet50.onnx"}.top_k(5);
for (const auto& r : results)
    std::cout << r.class_id << " " << r.score << " " << r.label << "\n";

// Detection — YOLO style (default) with NMS
auto detections = DnnDetector{"yolov8n.onnx"}
    .confidence_threshold(0.5f).labels(class_names)(frame);

// Detection — SSD style (two output blobs)
auto detections = DnnDetector{"ssd.onnx"}
    .style(DnnDetector::Style::SSD)
    .boxes_layer("detection_boxes")
    .scores_layer("detection_scores")(frame);

// Raw output tensor (for custom parsing)
std::vector<float> blob = DnnForward{"encoder.onnx"}(img);
```

**`DnnClassifier`** — runs `blobFromImage` → `forward` → `sortIdx`, returns top-K `ClassResult` sorted by score. Setters: `top_k(int)`, `input_size(int,int)`, `mean(cv::Scalar)`, `scale(float)`, `swap_rb(bool)`, `labels(vector<string>)`. Defaults: top_k=5, 224×224, scale=1/255, swap_rb=true.

**`DnnDetector`** — supports two output formats via `Style` enum. `Style::YOLO` (default): single blob `[1, N, 5+C]` (cx,cy,w,h,obj_conf,class_scores). `Style::SSD`: two blobs — boxes `[1,N,4]` as y1,x1,y2,x2 normalized + scores `[1,N,C]`. NMS applied via `cv::dnn::NMSBoxes`. Box coordinates are scaled back to original image dimensions. Defaults: 640×640, confidence_threshold=0.5, nms_threshold=0.4.

**`DnnForward`** — minimal wrapper: `blobFromImage` → `forward` → flatten to `std::vector<float>`. Use for models with non-standard output formats that require custom parsing. Defaults: 224×224, scale=1/255, swap_rb=true.

### Augmentation (`augmentation.hpp`)

Stochastic image augmentation ops for training data pipelines. All ops are header-only and templated on `Image<Format>`. Each op exposes two interfaces:
- `aug(img, rng)` — direct call with caller-owned `std::mt19937`
- `aug.bind(rng)` — returns a functor compatible with `operator|`

Format constraints are enforced at compile time via C++20 concepts (`AnyFormat`, `BGRFormat`).

```cpp
#include "improc/ml/augmentation.hpp"
using namespace improc::ml;
using namespace improc::core;

std::mt19937 rng(42);

// Standalone call
Image<BGR> flipped = RandomFlip{}.p(0.5f)(img, rng);

// Pipeline via .bind()
Image<BGR> result = img
    | RandomFlip{}.p(0.5f).bind(rng)
    | RandomBrightness{}.range(0.8f, 1.2f).bind(rng);

// Composition pipeline for training
auto augmentor = Compose<BGR>{}
    .add(RandomFlip{}.p(0.5f))
    .add(RandomRotate{}.range(-10.0f, 10.0f))
    .add(RandomApply<BGR>{ColorJitter{}.brightness(0.8f, 1.2f), 0.5f})
    .add(OneOf<BGR>{}
        .add(RandomGaussianNoise{}.std_dev(5.0f, 15.0f))
        .add(RandomSaltAndPepper{}.p(0.02f)));

Image<BGR> augmented = augmentor(img, rng);
```

**Geometric ops** (`RandomFlip`, `RandomRotate`, `RandomCrop`, `RandomResize`) — work on any `Image<Format>`. `RandomCrop` throws `ParameterError` if crop exceeds image size or dimensions not set.

#### Geometric Extras (v0.4.0)

**`RandomZoom`** — crops a random sub-region and resizes back to original size; simulates zoom-in.

```cpp
std::mt19937 rng(42);
Image<BGR> zoomed = RandomZoom{}.range(0.6f, 1.0f)(img, rng);
// or pipeline:
Image<BGR> zoomed2 = img | RandomZoom{}.range(0.7f, 0.9f).bind(rng);
```

Setters: `range(min_scale, max_scale)` — both in (0, 1], min ≤ max; default (0.7, 1.0).

**`RandomShear`** — applies a random shear transform via affine warp; borders filled with 0.

```cpp
Image<BGR> sheared = RandomShear{}.range(-15.0f, 15.0f)(img, rng);
Image<BGR> vshear  = RandomShear{}.range(-10.0f, 10.0f).axis(core::Axis::Vertical)(img, rng);
```

Setters: `range(min_deg, max_deg)` — min ≤ max; `axis(Axis)` — Horizontal (default) or Vertical.

**`RandomPerspective`** — randomly perturbs the four image corners and applies a homography warp.

```cpp
Image<BGR> warped = RandomPerspective{}.distortion_scale(0.5f)(img, rng);
```

Setter: `distortion_scale(s)` — in [0, 1]; each corner offset ≤ s × min(w,h)/2; default 0.5.

**Colour ops** (`RandomBrightness`, `RandomContrast`) — work on any `Image<Format>`. `ColorJitter` requires `Image<BGR>` (compile error on other formats).

#### Colour Extras (v0.4.0)

**`RandomGrayscale`** — converts BGR image to grayscale (3-channel gray) with probability `p`; Gray input always returned unchanged.

```cpp
Image<BGR> gray = RandomGrayscale{}.p(0.2f)(img, rng);
```

Setter: `p(prob)` — in [0, 1]; default 0.1.

**`RandomSolarize`** — inverts pixels at or above a threshold (efficient LUT); works on 8-bit types.

```cpp
Image<BGR> sol = RandomSolarize{}.threshold(128).p(0.5f)(img, rng);
```

Setters: `threshold(t)` — [0, 255]; `p(prob)` — [0, 1]; defaults: 128, 0.5.

**`RandomPosterize`** — reduces bits-per-channel via bitmasking (efficient LUT); works on 8-bit types.

```cpp
Image<BGR> post = RandomPosterize{}.bits(4).p(0.5f)(img, rng);
```

Setters: `bits(b)` — [1, 8]; `p(prob)` — [0, 1]; defaults: 4, 0.5.

**`RandomEqualize`** — histogram equalization with probability `p`; BGR: operates on Y channel in YCrCb; Gray: direct `cv::equalizeHist`.

```cpp
Image<BGR> eq = RandomEqualize{}.p(0.5f)(img, rng);
```

Setter: `p(prob)` — [0, 1]; default 0.5.

**Noise ops** (`RandomGaussianNoise`, `RandomSaltAndPepper`) — work on any `Image<Format>`. Clamp range adapts to pixel depth: `[0, 255]` for 8-bit, `[0, 1]` for float.

#### Erase / Dropout (v0.4.0)

**`RandomErasing`** — erases a randomly sampled rectangular region (fills with a constant value); up to 10 candidate rects sampled; falls back silently if none fits.

```cpp
Image<BGR> erased = RandomErasing{}.p(0.5f).scale(0.02f, 0.33f).ratio(0.3f, 3.3f).value(0)(img, rng);
```

Setters: `p(prob)` — [0, 1]; `scale(min, max)` — fraction of image area, 0 < min <= max <= 1; `ratio(min, max)` — aspect ratio, 0 < min <= max; `value(v)` — fill value [0, 255] for integer formats; for float formats the integer is used as-is as a float channel value. Defaults: p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.

**`GridDropout`** — divides the image into cells and independently zeros each with probability `ratio`.

```cpp
Image<BGR> dropped = GridDropout{}.ratio(0.5f).unit_size(32).value(0)(img, rng);
```

Setters: `ratio(r)` — (0, 1); `unit_size(s)` — pixels, must be > 0; `value(v)` — fill value [0, 255] for integer formats; for float formats the integer is used as-is as a float channel value. Defaults: ratio=0.5, unit_size=32, value=0.

**Composition ops** (`Compose<F>`, `RandomApply<F>`, `OneOf<F>`) — parameterised on `Format`, use `std::function` for type erasure. `OneOf` throws `AugmentError` if called with no augmentations added. `RandomApply` throws `ParameterError` if `p` is outside `[0, 1]`.

#### Bbox-aware geometric ops (`annotated.hpp`, `augment/bbox_compose.hpp`)

All 7 geometric ops accept `AnnotatedImage<Format>` in addition to `Image<Format>`. After each transform boxes are clipped to image bounds; boxes where `clipped_area / original_area < min_area_ratio` (default 0.1) are dropped.

```cpp
#include "improc/ml/augmentation.hpp"

struct BBox {
    cv::Rect2f  box;       // pixel-coordinate annotation
    int         class_id;  // default 0
    std::string label;
};

template<AnyFormat Format>
struct AnnotatedImage { Image<Format> image; std::vector<BBox> boxes; };

// Each geometric op gains a second operator():
AnnotatedImage<BGR> r = RandomFlip{}.p(0.5f)(ann, rng);
AnnotatedImage<BGR> r = RandomRotate{}.range(-15.f, 15.f)(ann, rng);
AnnotatedImage<BGR> r = RandomCrop{}.width(224).height(224)(ann, rng);
// … same for RandomResize, RandomZoom, RandomShear, RandomPerspective

// Tune the drop threshold per-op:
RandomCrop{}.min_area_ratio(0.3f).width(200).height(200)(ann, rng);

// BBoxCompose<F> — sequential bbox-aware pipeline with bind(rng) / operator|:
BBoxCompose<BGR> pipeline;
pipeline
    .add([](auto a, auto& r){ return RandomFlip{}.p(0.5f)(std::move(a), r); })
    .add([](auto a, auto& r){ return RandomRotate{}.range(-15.f,15.f)(std::move(a), r); });

auto result = pipeline(ann, rng);           // direct call
auto result = ann | pipeline.bind(rng);     // pipeline form
```

#### Blur Extras (v0.4.0)

**`RandomBlur`** — randomly applies one of Gaussian / Median / Bilateral blur with a random odd kernel size.

```cpp
// All three types (default)
Image<BGR> blurred = RandomBlur{}.kernel_size(3, 11)(img, rng);

// Gaussian only (safe for Float32)
Image<BGR> g = RandomBlur{}.types({RandomBlur::Type::Gaussian}).kernel_size(3, 7)(img, rng);
```

Setters: `types(vector<Type>)` — non-empty subset of {Gaussian, Median, Bilateral}; `kernel_size(min_k, max_k)` — both odd, in [3, 31], min <= max; default all three types, kernel (3, 7). **Note:** Bilateral throws `ParameterError` at call time for Float32/Float32C3 inputs.

**`RandomSharpness`** — unsharp-mask sharpening (`out = img + strength * (img - GaussianBlur(img, 3))`); applied with probability `p`.

```cpp
Image<BGR> sharp = RandomSharpness{}.range(0.5f, 1.5f).p(0.7f)(img, rng);
```

Setters: `range(min_s, max_s)` — 0 <= min <= max; `p(prob)` — [0, 1]; defaults: range=(0, 1), p=0.5.

#### MixUp / CutMix (v0.4.0-C)

`LabeledImage<F>` pairs an image with a soft label vector (`std::vector<float>`). `MixUp` and `CutMix` are binary ops — they accept two `LabeledImage<F>` values and return a blended result. λ is sampled from Beta(α, α).

```cpp
#include "improc/ml/augmentation.hpp"
using namespace improc::ml;
using namespace improc::core;

std::mt19937 rng(42);

// 3-class one-hot labels
LabeledImage<BGR> a{img_a, {1.0f, 0.0f, 0.0f}};
LabeledImage<BGR> b{img_b, {0.0f, 1.0f, 0.0f}};

// Standalone ops
auto mu = MixUp{}.alpha(0.4f)(a, b, rng);   // λ·a + (1-λ)·b
auto cm = CutMix{}.alpha(1.0f)(a, b, rng);  // paste rect from b; label by area ratio

// MixCompose — sequential binary pipeline; secondary stays fixed
MixCompose<BGR> pipe;
pipe.add([](auto a, const auto& b, auto& r){ return MixUp{}.alpha(0.4f)(std::move(a), b, r); });

auto result  = pipe(a, b, rng);        // direct call
auto result2 = a | pipe.bind(b, rng);  // pipeline form
```

**`MixUp`** setters: `alpha(a)` — Beta parameter, must be > 0 (default 0.4); `p(prob)` — [0, 1] (default 1.0). Throws `ParameterError` if image sizes differ or label vectors are empty/mismatched.

**`CutMix`** setters: `alpha(a)` — Beta parameter, must be > 0 (default 1.0); `p(prob)` — [0, 1] (default 1.0). Patch size: `w = W·sqrt(1-λ)`, `h = H·sqrt(1-λ)`; actual λ = 1 − (w·h)/(W·H). Throws `ParameterError` if image sizes differ or label vectors are empty/mismatched.

**`MixCompose<F>`** chains binary ops sequentially — primary transforms through each op, secondary stays fixed. `bind(secondary, rng)` returns an `operator|`-compatible unary functor. Empty composer returns primary unchanged. Throws `ParameterError` for null ops.

#### VOC Dataset Loader (v0.4.0-D)

`VocDataset` loads Pascal VOC annotation datasets into `AnnotatedImage<BGR>` train/val/test splits. `parse_voc_xml` parses a single annotation file and can be used independently.

```cpp
#include "improc/ml/ml.hpp"
using namespace improc::ml;

// Full dataset loading
VocDataset ds;
ds.classes({"cat", "dog"})   // optional — fixes id order and filters others
  .skip_difficult(true)       // default
  .val_ratio(0.1f)
  .test_ratio(0.2f)
  .shuffle_seed(42);          // for random-split reproducibility

auto ok = ds.load_from_directory("data/VOC2012");
if (!ok) { std::cerr << ok.error().message; }

for (const auto& ann : ds.train()) {
    // ann.image: Image<BGR>
    // ann.boxes: std::vector<BBox>  (each has .box, .class_id, .label)
}
ds.class_name_for(0);  // "cat"

// Parse a single XML file
std::unordered_map<std::string, int> class_map;
auto ann = parse_voc_xml("Annotations/000001.xml", "JPEGImages/", class_map);
```

**Split detection:** If `<root>/ImageSets/Main/train.txt` exists, VOC split files are used. Otherwise all XMLs are randomly split by `val_ratio`/`test_ratio`.

**`VocDataset` setters:** `classes(list)`, `skip_difficult(bool)`, `test_ratio(f)`, `val_ratio(f)`, `shuffle_seed(n)`.  
**`parse_voc_xml` params:** `xml_path`, `images_dir`, `class_map&`, `skip_difficult=true`, `filter_unknown=false`.

#### COCO Dataset Loader (v0.4.0-E)

`CocoDataset` loads COCO JSON annotation datasets into `AnnotatedImage<BGR>` splits. Each split is loaded independently via explicit `load_train`/`load_val`/`load_test` calls. Class mapping is shared across all splits and auto-built or user-supplied via `.classes()`. `parse_coco_json` parses a single COCO JSON file and can be used independently.

```cpp
#include "improc/ml/ml.hpp"
using namespace improc::ml;

// Full dataset loading
CocoDataset ds;
ds.classes({"cat", "dog"})  // optional — fixes id order and filters others; call before load_*
  .skip_crowd(true);         // default

auto ok = ds.load_train("annotations/instances_train2017.json", "images/train2017/");
if (!ok) { std::cerr << ok.error().message; }

ok = ds.load_val("annotations/instances_val2017.json", "images/val2017/");
if (!ok) { std::cerr << ok.error().message; }

for (const auto& ann : ds.train()) {
    // ann.image: Image<BGR>
    // ann.boxes: std::vector<BBox>  (each has .box, .class_id, .label)
}
ds.class_name_for(0);  // "cat"

// Parse a single COCO JSON file
std::unordered_map<std::string, int> class_map;
auto items = parse_coco_json("instances_train.json", "images/", class_map);
```

**`CocoDataset` setters:** `classes(list)` (MUST call before first load_\*), `skip_crowd(bool)` (default true).  
**`parse_coco_json` params:** `json_path`, `images_dir`, `class_map&`, `skip_crowd=true`, `filter_unknown=false`.  
**COCO id remapping:** COCO category IDs are non-contiguous (e.g. 1, 2, …90 with gaps). They are remapped to 0-indexed sequential IDs in encounter order (or user-specified order via `.classes()`).

#### Segmentation Types (`segmented.hpp`, v0.4.0-F)

`SegmentedImage<F>` holds an image, a per-pixel class mask, and an optional instance mask.

```cpp
template<AnyFormat Format>
struct SegmentedImage {
    Image<Format>              image;
    Image<Gray>                class_mask;    // pixel = class_id; 255 = void (kept as-is)
    std::optional<Image<Gray>> instance_mask; // nullopt when not loaded
};

// operator| pipeline support
SegmentedImage<BGR> result = seg | RandomFlip{}.p(0.5f).bind(rng);
```

#### Segmentation Augmentation (v0.4.0-F/G)

All geometric ops (`RandomFlip`, `RandomRotate`, `RandomCrop`, `RandomResize`, `RandomZoom`, `RandomShear`, `RandomPerspective`) accept `SegmentedImage<F>` — same spatial transform applied to image (INTER_LINEAR) and both masks (INTER_NEAREST). Color ops (`RandomBrightness`, `RandomContrast`, `ColorJitter`, `RandomGaussianNoise`, `RandomSaltAndPepper`, `RandomBlur`) apply to image only; masks pass through unchanged.

`SegCompose<F>` — sequential composer, mirrors `BBoxCompose<F>`:
```cpp
SegCompose<BGR> pipeline;
pipeline.add([](SegmentedImage<BGR> s, std::mt19937& r) { return RandomFlip{}.p(0.5f)(std::move(s), r); });
pipeline.add([](SegmentedImage<BGR> s, std::mt19937& r) { return ColorJitter{}(std::move(s), r); });
auto out = pipeline(seg, rng);
// or: seg | pipeline.bind(rng)
```

#### VOC Segmentation Dataset Loader (`voc_seg_dataset.hpp`, v0.4.0-H)

`VocSegDataset` loads VOC segmentation directories into `SegmentedImage<BGR>` train/val/test splits.

```cpp
#include "improc/ml/ml.hpp"
using namespace improc::ml;

VocSegDataset ds;
ds.classes({"background","aeroplane","bicycle","bird","boat",
            "bottle","bus","car","cat","chair","cow",
            "diningtable","dog","horse","motorbike","person",
            "pottedplant","sheep","sofa","train","tvmonitor"})
  .load_instance_masks(true);

auto ok = ds.load_from_directory("data/VOC2012");
if (!ok) { std::cerr << ok.error().message; return 1; }

for (const auto& seg : ds.train()) {
    // seg.image: Image<BGR>
    // seg.class_mask: Image<Gray> (pixel = class_id, 255 = void)
    // seg.instance_mask: std::optional<Image<Gray>>
}
ds.class_name_for(15);  // "person"

// Parse a single entry
auto r = parse_voc_seg("2007_000032", "JPEGImages/", "SegmentationClass/", {});
```

**Directory structure:** `JPEGImages/`, `SegmentationClass/` (required), `SegmentationObject/` (optional, for instance masks), `ImageSets/Segmentation/train.txt` and `val.txt` (optional; falls back to 10%/10% random split).

**Setters:** `classes(vector)` (index = class_id → name), `load_instance_masks(bool)` (default false).

#### Multi-Object Tracking (v0.5.0)

**Umbrella header:** `#include "improc/ml/tracking/tracking.hpp"`

**Core types** (`track.hpp`):

```cpp
struct Track {
    int   id           = -1;   // persistent track ID
    BBox  bbox;                // current bbox (pixel coords)
    float confidence   = 0.0f;
    int   age          = 0;    // frames since last detection match
    bool  is_confirmed = false;
};

struct TrackGT {               // ground-truth annotation for evaluation
    int  id;
    BBox bbox;
};

// Concept satisfied by all tracker structs
template<typename T>
concept TrackerType = requires(T t, std::vector<Detection> d) {
    { t.update(d) } -> std::same_as<std::vector<Track>>;
    { t.reset()   } -> std::same_as<void>;
};
```

All three trackers satisfy `TrackerType` and are drop-in replaceable.

---

**`IouTracker`** (`iou_tracker.hpp`) — Greedy IoU matching with age-based culling. No motion model; suitable for fast, static, or camera-fixed scenes.

| Setter | Default | Description |
|---|---|---|
| `min_iou(float)` | 0.3 | Minimum IoU to match a detection to a track |
| `max_age(int)` | 1 | Frames to keep a track alive without a match |

```cpp
IouTracker tracker;
tracker.min_iou(0.3f).max_age(2);
std::vector<Track> tracks = tracker.update(detections);
tracker.reset();
```

---

**`SortTracker`** (`sort_tracker.hpp`) — SORT algorithm: Hungarian assignment on (1 − IoU) cost, constant-velocity Kalman filter for motion prediction, confirmation gate.

| Setter | Default | Description |
|---|---|---|
| `max_age(int)` | 3 | Frames to keep an unmatched track alive |
| `min_hits(int)` | 3 | Hits required before a track is reported as confirmed |
| `iou_threshold(float)` | 0.3 | Minimum IoU to accept a Hungarian assignment |

```cpp
SortTracker tracker;
tracker.max_age(3).min_hits(3).iou_threshold(0.3f);
std::vector<Track> tracks = tracker.update(detections);  // only confirmed tracks returned
tracker.reset();
```

`Track::age` is the number of frames since the last detection match. `Track::is_confirmed` is `true` once `hits >= min_hits`. Tracks with `age > max_age` are removed.

---

**`ByteTracker`** (`byte_tracker.hpp`) — BYTE algorithm: two-stage matching that recovers occluded tracks via low-confidence detections.

- **Stage 1** — Hungarian on high-confidence detections (`confidence >= high_conf_threshold`) vs all existing tracklets.
- **Stage 2** — Greedy IoU on low-confidence detections (`low_conf_threshold <= confidence < high_conf_threshold`) vs unmatched tracklets from Stage 1.

| Setter | Default | Description |
|---|---|---|
| `max_age(int)` | 3 | Frames to keep an unmatched track alive |
| `min_hits(int)` | 3 | Hits required before a track is confirmed |
| `high_conf_threshold(float)` | 0.6 | Stage 1 threshold |
| `low_conf_threshold(float)` | 0.1 | Detections below this are discarded |

```cpp
ByteTracker tracker;
tracker.max_age(3).min_hits(3)
       .high_conf_threshold(0.6f).low_conf_threshold(0.1f);
std::vector<Track> tracks = tracker.update(detections);  // only confirmed tracks returned
tracker.reset();
```

---

**`TrackingEval`** / **`TrackingMetrics`** (`tracking_eval.hpp`) — Frame-by-frame accumulator for standard MOT metrics.

```cpp
struct TrackingMetrics {
    float MOTA = 0.0f;  // 1 − (FN + FP + IDSW) / GT_total
    float MOTP = 0.0f;  // mean IoU of matched pairs
    float IDF1 = 0.0f;  // 2·IDTP / (2·IDTP + IDFP + IDFN)
    int   FP   = 0;
    int   FN   = 0;
    int   IDSW = 0;
};

TrackingEval eval;
eval.iou_threshold(0.5f);           // default 0.5

for (int f = 0; f < n_frames; ++f)
    eval.update(predicted_tracks[f], ground_truth[f]);

TrackingMetrics m = eval.compute(); // [[nodiscard]]
eval.reset();                       // clears all accumulated state
```

`update()` uses globally-greedy IoU matching (all valid pairs sorted by IoU descending) — ties are broken by insertion order. An ID switch is counted when a GT is matched to a different track ID than in the previous frame. IDF1 is computed via global bipartite matching on co-occurrence counts (Hungarian).

---

## `improc::threading` — Concurrency utilities

**Status: Implemented**

### `ThreadPool` (`threading/thread_pool.hpp`)

General-purpose thread pool. Workers drain a task queue; `submit` returns a typed future; `submit_detached` is fire-and-forget. Destructor drains the queue and joins all workers (graceful shutdown). Non-copyable, non-movable.

```cpp
ThreadPool pool(4);                                          // 4 workers (default: hardware_concurrency)

auto future = pool.submit([](int a, int b){ return a+b; }, 3, 4);
int result  = future.get();                                  // 7

pool.submit_detached([]{ heavy_work(); });                   // fire-and-forget
```

Throws `ParameterError` if constructed with 0 threads. Exceptions from tasks propagate through `future.get()`.

### `FramePipeline<Result>` (`threading/frame_pipeline.hpp`)

Header-only template. Connects `CameraCapture` to `ThreadPool` for typed frame-by-frame processing. Holds references (not ownership) to both. Non-copyable, non-movable.

```cpp
improc::io::CameraCapture camera(0);
ThreadPool pool(4);
FramePipeline<cv::Mat> pipeline(camera, pool);

pipeline.start([](cv::Mat frame) {
    return frame | Resize{}.width(224).height(224) | ToFloat32C3{};
});

while (running) {
    if (auto result = pipeline.tryPop()) {
        // use *result (cv::Mat)
    }
}
pipeline.stop();
```

`tryPop()` returns `std::optional<Result>` — `std::nullopt` if no result is ready or `start()` was not called. `start()` called twice throws `improc::Exception`. `stop()` is idempotent.

`FramePipeline<void>` is not supported (`std::optional<void>` is not valid C++). For fire-and-forget frame processing use `pool.submit_detached(processor, camera.getFrame())` directly.

---

## `improc::visualization` — Chart and display utilities

**Status: Implemented**

All chart functors return `Image<BGR>` and compose with the existing `operator|` pipeline. No external plotting library — everything uses OpenCV.

### `Histogram` (`visualization/histogram.hpp`)

Pipeline op. Computes per-channel histograms and renders them as curves on a black canvas. Accepts `Image<BGR>`, `Image<Gray>`, or `Image<Float32>`.

```cpp
#include "improc/visualization/visualization.hpp"
using namespace improc::visualization;

Image<BGR> hist  = bgr_image | Histogram{};                          // 512×256, 256 bins
Image<BGR> hist2 = gray_image | Histogram{}.bins(64).width(400).height(200);
Image<BGR> hist3 = f32_image  | Histogram{};                         // auto min-max range
```

BGR input renders three colored curves (blue, green, red). Gray/Float32 input renders a single white curve.

### `LinePlot` (`visualization/line_plot.hpp`)

Standalone functor. Takes a `std::vector<float>` and renders a connected polyline scaled to the canvas.

```cpp
std::vector<float> loss = {1.2f, 0.9f, 0.7f, 0.5f, 0.3f};
Image<BGR> plot = LinePlot{}.title("Train Loss").color({0, 200, 255})(loss);
```

Values are min-max normalised to canvas height. A single value or all-equal values renders a horizontal centre line. Throws `ParameterError` if vector is empty.

### `Scatter` (`visualization/scatter.hpp`)

Standalone functor. Takes `xs` and `ys` vectors and renders filled circles, each axis independently normalised.

```cpp
Image<BGR> sc = Scatter{}.title("Features").color({0, 255, 255}).point_radius(5)(xs, ys);
```

Throws `ParameterError` if either vector is empty or `xs.size() != ys.size()`. An 8 px margin ensures extreme points are fully visible.

### `Show` (`visualization/show.hpp`, header-only)

Passthrough display op. Calls `cv::imshow()` + `cv::waitKey()` and returns the image unchanged — enables inline display anywhere in a pipeline.

```cpp
// Block until key press (default)
bgr_image | Histogram{} | Show{"Histogram"};

// Non-blocking — suitable for camera loops
result | Show{"Camera"}.wait_ms(1);

// Stand-alone chart display
LinePlot{}.title("Loss")(loss) | Show{"Loss"};
```

`Show` accepts only `Image<BGR>`. To display Gray/Float32 images convert first or pipe through `Histogram{}`.

### `DrawBoundingBoxes` (`visualization/draw.hpp`, header-only)

Draws detection boxes (and optional labels / confidence scores) onto a clone of the source image. Takes a `std::vector<improc::ml::Detection>` at construction and returns `Image<BGR>` — pipeline-compatible.

```cpp
#include "improc/visualization/draw.hpp"
using namespace improc::visualization;

std::vector<improc::ml::Detection> dets = detector(frame);

// Annotate with default settings (green boxes, thickness 2, label + confidence)
Image<BGR> annotated = frame | DrawBoundingBoxes{dets};

// Custom appearance
Image<BGR> annotated = frame | DrawBoundingBoxes{dets}
    .color({0, 0, 255})      // red
    .thickness(1)
    .font_scale(0.4)
    .show_label(true)
    .show_confidence(false);

// Full inference + annotation pipeline
Image<BGR> result = frame
    | Resize{}.width(640).height(640)
    | DrawBoundingBoxes{detector(frame)}.thickness(2)
    | Show{"Detections"}.wait_ms(1);
```

`DrawBoundingBoxes` draws onto a **clone** — the source image is never modified. Throws `ParameterError` if `thickness <= 0` or `font_scale <= 0`.

### `Montage` (`visualization/montage.hpp`)

Arranges a `vector<Image<BGR>>` into a grid. Setters: `.cols(int)`, `.cell_size(int w, int h)`, `.gap(int)`, `.background(cv::Scalar)`. Returns a single tiled `Image<BGR>`.

### Umbrella include

```cpp
#include "improc/visualization/visualization.hpp"  // includes all five
```

---

## `improc::onnx` — ONNX Runtime inference

**Status: Implemented** (ONNX Runtime 1.20.1 — CPU + CoreML EP on Apple Silicon)

**Umbrella header:** `improc/onnx/onnx.hpp`

Downloaded automatically via CMake FetchContent — no separate installation required. On Apple Silicon the CoreML execution provider is registered for compatible ops with transparent CPU fallback.

### `TensorInfo` (`onnx_session.hpp`)

Exchange type between `OnnxSession` and its callers. Carries name, shape, and flat float data.

```cpp
struct TensorInfo {
    std::string          name;   // input/output node name
    std::vector<int64_t> shape;  // e.g. {1, 3, 224, 224}
    std::vector<float>   data;   // flat row-major floats
};
```

### `OnnxSession` (`onnx_session.hpp`)

Thin wrapper over `Ort::Session`. ORT types are fully hidden behind a pimpl — including `onnx_session.hpp` does not require `onnxruntime_cxx_api.h` in caller code.

```cpp
OnnxSession session;
auto err = session.load("model.onnx");  // std::expected<void, Error>
if (!err) { std::cerr << err.error().message; }

auto outputs = session.run({{session.input_names()[0], {1,3,224,224}, data}});
// returns std::expected<std::vector<TensorInfo>, Error>

session.input_names();   // std::vector<std::string>
session.output_names();  // std::vector<std::string>
session.is_loaded();     // bool
```

`load()` returns `Error{Code::InvalidModelFile}` for a missing file or wrong extension; `Error{Code::OnnxModelLoadFailed}` if ORT rejects the model. `run()` returns `Error{Code::OnnxSessionNotLoaded}` if called before `load()`, or `Error{Code::OnnxInferenceFailed}` on ORT error.

`OnnxSession` is **non-copyable, movable**.

### `OnnxClassifier` (`onnx_classifier.hpp`)

Wraps `OnnxSession` with a complete image preprocessing pipeline and top-K post-processing. Mirrors the `DnnClassifier` fluent API.

```cpp
OnnxClassifier cls{"mobilenet.onnx"};
cls.input_size(224, 224)
   .mean(0.485f, 0.456f, 0.406f)   // B, G, R order
   .scale(1.0f / 255.0f)
   .swap_rb(true)                   // BGR → RGB before inference
   .labels(class_names)
   .top_k(5);

auto result = cls(img);  // std::expected<std::vector<ClassResult>, Error>
```

Expects the model's first output to be a 1-D score vector `[1, N]`. Results are sorted by score descending. Constructor throws `ModelError` if the model cannot be loaded.

| Setter | Default | Throws |
|---|---|---|
| `top_k(int k)` | 5 | `ParameterError` if k ≤ 0 |
| `input_size(int w, int h)` | 224 × 224 | `ParameterError` if either ≤ 0 |
| `mean(float b, float g, float r)` | 0, 0, 0 | — |
| `scale(float s)` | 1/255 | `ParameterError` if s ≤ 0 |
| `swap_rb(bool)` | `true` | — |
| `labels(vector<string>)` | empty | — |

### `OnnxDetector` (`onnx_detector.hpp`)

Wraps `OnnxSession` with image preprocessing, format-aware YOLO/SSD output parsing, and NMS. Mirrors the `DnnDetector` fluent API.

```cpp
OnnxDetector det{"yolov8n.onnx"};
det.input_size(640, 640)
   .confidence_threshold(0.5f)
   .nms_threshold(0.4f)
   .labels(class_names);

auto boxes = det(frame);  // std::expected<std::vector<Detection>, Error>
```

Two output formats are supported via `Style`:

| Style | Output layout | Notes |
|---|---|---|
| `Style::YOLO` (default) | `[1, N, 5+C]` (YOLOv5) or `[1, 4+C, N]` (YOLOv8) | Auto-detected by shape heuristic |
| `Style::SSD` | boxes `[1,N,4]` (y1,x1,y2,x2 norm.) + scores `[1,N,C]` | Two output tensors required |

YOLOv8 ONNX export via `model.export(format="onnx")` produces `[1, 4+C, N]` — handled automatically. Box coordinates are rescaled to the original image dimensions before NMS.

| Setter | Default | Throws |
|---|---|---|
| `style(Style)` | `YOLO` | — |
| `confidence_threshold(float)` | 0.5 | `ParameterError` if outside [0,1] |
| `nms_threshold(float)` | 0.4 | `ParameterError` if outside [0,1] |
| `input_size(int w, int h)` | 640 × 640 | `ParameterError` if either ≤ 0 |
| `mean(float b, float g, float r)` | 0, 0, 0 | — |
| `scale(float s)` | 1/255 | `ParameterError` if s ≤ 0 |
| `swap_rb(bool)` | `true` | — |
| `labels(vector<string>)` | empty | — |

---

## `improc::views` — Lazy Image Pipeline

**Header:** `#include "improc/views/views.hpp"`

Lazy, ranges-style pipeline layer. Unlike the eager `operator|` in `improc::core`,
adapters from `improc::views` build a deferred computation chain that executes only
when `views::to<T>()` is called — avoiding intermediate `Image<F>` copies.

### M1 — Single Image

| Symbol | Description |
|---|---|
| `views::transform(op)` | Wraps any `improc::core` op into a lazy adapter; no computation at this point |
| `views::to<Image<F>>()` | Materializes the lazy chain into `Image<F>`; triggers the full op sequence |

Format mismatch between the view and `to<>()` is a **compile error** — no runtime check needed.

**Example:**
```cpp
#include "improc/views/views.hpp"
namespace views = improc::views;
using namespace improc::core;

auto view = img
    | views::transform(Resize{}.width(224).height(224))
    | views::transform(GaussianBlur{}.kernel_size(3))
    | views::transform(Brightness{}.delta(10.0));

Image<BGR> result = view | views::to<Image<BGR>>();
```

### M2 — In-Memory Collections (`std::vector<Image<F>>`)

| Symbol | Description |
|---|---|
| `views::transform(op)` | Applies op lazily to each collection element |
| `views::filter(pred)` | Skips elements where `pred(img)` returns `false` |
| `views::take(n)` | Stops iteration after `n` elements |
| `views::drop(n)` | Skips the first `n` elements |
| `views::to<std::vector<Image<F>>>()` | Collects all remaining elements into a vector |

**Example:**
```cpp
std::vector<Image<BGR>> images = load_dataset("path/");

auto batch = images
    | views::transform(Resize{}.width(224).height(224))
    | views::filter([](const Image<BGR>& img) { return img.cols() > 0; })
    | views::drop(10)
    | views::take(32)
    | views::to<std::vector<Image<BGR>>>();

// Or iterate lazily — one image processed at a time
for (const auto& img : images | views::transform(Resize{}.width(224).height(224))) {
    process(img);
}
```

### M3 — External Sources

| Symbol | Description |
|---|---|
| `views::VideoView{reader}` | Lazy single-pass wrapper over a `VideoReader` (non-owning; reader must outlive view) |
| `views::from_dir(path, exts)` | Lazy view over matching image files in a directory; throws `FileNotFoundError` if path missing |

Both sources compose with all M2 adapters (`transform`, `filter`, `take`, `drop`) and materialize via `views::to<std::vector<Image<BGR>>>()`.

**Examples:**
```cpp
// VideoReader source — frames processed one at a time, O(1) RAM
improc::io::VideoReader reader{"clip.mp4"};
auto frames = views::VideoView{reader}
    | views::transform(Resize{}.width(640).height(640))
    | views::take(100)
    | views::to<std::vector<Image<BGR>>>();

// Or iterate lazily (no materialization)
for (const auto& frame : views::VideoView{reader} | views::take(100)) {
    process(frame);
}

// Directory source — loads files one at a time
auto batch = views::from_dir("dataset/train/cats/", {".jpg", ".png"})
    | views::transform(Resize{}.width(224).height(224))
    | views::filter([](const Image<BGR>& img) { return !img.mat().empty(); })
    | views::to<std::vector<Image<BGR>>>();
```

### M4 — Advanced Adapters

| Symbol | Description |
|---|---|
| `views::batch(n)` | Groups N consecutive elements into `std::vector<Image<F>>` chunks. Last chunk may be smaller. |
| `views::enumerate` | Pairs each element with a zero-based `std::size_t` index. No parentheses — tag object. |
| `views::zip(v1, v2)` | Pairs elements from two sources; stops at the shorter. Free function, not `operator\|`. |

Works with all M1–M3 source types: `std::vector<Image<F>>`, `VideoView`, and `from_dir()`.

```cpp
namespace views = improc::views;
using namespace improc::core;

std::vector<Image<BGR>> images = load_dataset("train/");
std::vector<Image<BGR>> masks  = load_dataset("masks/");

// batch — ML training loop
for (const auto& mini_batch : images | views::transform(Resize{}.width(224).height(224))
                                     | views::batch(32))
    model.train(mini_batch);

// enumerate — index tracking
for (const auto& [idx, img] : images | views::enumerate)
    std::cout << std::format("frame {}: {}x{}\n", idx, img.cols(), img.rows());

// zip — image + mask pairs
for (const auto& [img, mask] : views::zip(images, masks))
    result.push_back(img | ApplyMask{}.mask(mask));
```

---

## Planned namespaces

| Namespace | Purpose |
|---|---|
| `improc::cuda` | Wrapper over `cv::cuda` for GPU-accelerated ops |
