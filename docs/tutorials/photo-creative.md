# Photo and Creative Effects

`improc++` v0.10.0 adds nine ops for non-photorealistic rendering, temporal denoising, high-dynamic-range imaging, and panorama stitching. These ops belong to three conceptual groups: artistic filters that transform a single image into a stylised or enhanced version (EdgePreservingFilter, DetailEnhance, Stylize, PencilSketch, SeamlessClone), multi-image ops that combine two or more frames into a composite (NLMeansDenoisingMulti, MergeHDR), and finalisation ops that prepare images for display or storage (ToneMap, Stitch).

All nine ops are available via the umbrella include `#include "improc/core/pipeline.hpp"` inside the `improc::core` namespace. The single-image artistic ops (`EdgePreservingFilter`, `DetailEnhance`, `Stylize`) integrate seamlessly into `operator|` pipelines; the multi-image and struct-returning ops are called directly. No OpenCV contrib module is required — every op in this group uses standard `cv::photo`, `cv::stitching`, and `cv::hdr` functionality.

Most parameters follow the domain-transform conventions established in OpenCV's photo module: `sigma_s` controls the spatial extent of a filter (typically measured in pixels, valid range 1–200), while `sigma_r` controls sensitivity to colour differences (valid range 0–1). Keeping `sigma_s` large and `sigma_r` small produces stronger stylisation with sharper colour boundaries; increasing `sigma_r` allows the filter to cross more colour edges, producing a flatter, paint-like result.

The multi-image ops (temporal denoising, HDR) expect a `std::vector<Image<BGR>>`. Inputs must all have the same pixel dimensions; mismatched sizes cause a runtime error from the underlying OpenCV call. The `PencilSketch` and `Stitch` ops return dedicated result structs (`PencilSketchResult` and `StitchResult`) rather than a single `Image<>`, so they cannot be used directly inside a pipeline chain.

## Prerequisites
- Completed [Building a Pipeline](building-a-pipeline.md)
- `improc/core/pipeline.hpp`

## Edge-Preserving Filter

`EdgePreservingFilter` smooths texture detail while keeping edges crisp and well-defined. It supports two underlying algorithms: the recursive filter (`Filter::Recursive`, the default) is fast and well-suited for real-time use; the normalised convolution variant (`Filter::NormConv`) is slightly more accurate at the cost of higher computation. Both share the same `sigma_s` / `sigma_r` parameterisation.

```cpp
#include "improc/core/pipeline.hpp"
using namespace improc::core;

Image<BGR> src = /* load or synthesize */;

// EdgePreservingFilter — smooths texture while preserving edges
Image<BGR> ep_rec = src | EdgePreservingFilter{}
    .sigma_s(60.f)                                    // spatial scale (default: 60)
    .sigma_r(0.4f)                                    // range scale (default: 0.4)
    .filter(EdgePreservingFilter::Filter::Recursive);  // Recursive (default) or NormConv

Image<BGR> ep_nc = src | EdgePreservingFilter{}
    .sigma_s(60.f)
    .sigma_r(0.4f)
    .filter(EdgePreservingFilter::Filter::NormConv);
```

## Detail Enhance

`DetailEnhance` amplifies fine structural details (thin lines, textures, edges) while suppressing noise in flat regions. It works by computing a detail layer and boosting it relative to a smoothed base, producing a result that looks sharper than the input without introducing halos. The recommended `sigma_s` is much smaller than for stylisation ops (around 10 pixels) because the goal is to preserve micro-structure rather than to generalise across regions.

```cpp
// DetailEnhance — sharpens fine detail while smoothing flat regions
Image<BGR> detail = src | DetailEnhance{}
    .sigma_s(10.f)    // spatial scale (default: 10)
    .sigma_r(0.15f);  // range scale (default: 0.15)
```

## Stylize

`Stylize` produces an oil-painting-like artistic effect by quantising colours within edge-bounded regions. Internally it is built on the same domain-transform framework as `EdgePreservingFilter`, but applies additional colour clustering to achieve the characteristic flat-colour, painterly look. Use it as a drop-in replacement for `EdgePreservingFilter` when the goal is creative expression rather than faithful edge preservation.

```cpp
// Stylize — oil-painting artistic effect
Image<BGR> styled = src | Stylize{}
    .sigma_s(60.f)    // spatial scale (default: 60)
    .sigma_r(0.45f);  // range scale (default: 0.45)
```

## Pencil Sketch

`PencilSketch` decomposes an image into pencil-like stroke lines and returns two complementary outputs via `PencilSketchResult`: a grayscale sketch (`sk.gray`) emphasising edges and a colour version (`sk.color`) that overlays those strokes on a tone-mapped rendering of the original. The `shade_factor` parameter controls the brightness of the pencil shading — low values (near 0) give sparse, dark lines while values approaching 0.1 produce denser, lighter shading.

```cpp
// PencilSketch — returns both a gray sketch and a colour version
PencilSketchResult sk = PencilSketch{}
    .sigma_s(60.f)          // spatial scale (default: 60)
    .sigma_r(0.07f)         // range scale (default: 0.07)
    .shade_factor(0.05f)    // brightness of pencil shading (default: 0.05)
    (src);

Image<Gray> sketch_gray  = sk.gray;   // grayscale pencil lines
Image<BGR>  sketch_color = sk.color;  // colour version
```

## Seamless Clone

`SeamlessClone` composites a source patch into a destination image using Poisson blending so that the boundary between the two regions is invisible. The caller provides a white mask (pixel value 255) covering the region of `src` that should be blended into `dst`, plus a `center` point in destination coordinates where the patch will be placed. Three blending modes are available: `Normal` preserves the source's colour and texture, `Mixed` blends gradients from both source and destination (useful when the background texture should show through), and `Monochrome` is intended for single-channel compositing. The call throws `ParameterError` if the center point falls outside the destination bounds.

```cpp
// SeamlessClone — Poisson-blending compositing
// src: object to paste; dst: background; mask: white (255) = blend region
cv::Mat mask(src.mat().size(), CV_8U, cv::Scalar(255));
cv::Point center{dst.mat().cols / 2, dst.mat().rows / 2};

Image<BGR> cloned = SeamlessClone{}
    .mode(SeamlessClone::Mode::Normal)   // Normal, Mixed, or Monochrome (default: Normal)
    (src, dst, Image<Gray>{mask}, center);
// Throws ParameterError if center is outside dst bounds
```

## Temporal Denoising (NLMeansDenoisingMulti)

`NLMeansDenoisingMulti` removes noise from a video sequence by comparing patches across multiple frames rather than within a single frame. This temporal approach is significantly more effective than single-frame denoising because it exploits the redundancy of nearly static content across frames. The op denoises the centre frame of the supplied vector; the `temporal_window_size` must be odd and must match the length of the input vector. At least two frames are required; passing a single-element vector raises an error.

```cpp
// NLMeansDenoisingMulti — temporal denoising over a sequence of frames
// Requires >= 2 frames; denoises the "centre" frame using neighbouring frames
std::vector<Image<BGR>> frames = { frame_t_minus1, frame_t, frame_t_plus1 };

Image<BGR> denoised = NLMeansDenoisingMulti{}
    .h(3.f)                    // filter strength (default: 3)
    .temporal_window_size(3)   // number of frames (must be odd, default: 3)
    (frames);
```

## HDR Pipeline (MergeHDR + ToneMap)

`MergeHDR` combines a bracketed exposure sequence into a single high-dynamic-range image stored as `Image<Float32C3>`. The `Mertens` method performs exposure fusion: it computes a weighted blend of the exposure stack that emphasises well-exposed regions in each frame, and requires no exposure time metadata. The `Debevec` method implements true radiometric HDR reconstruction and requires the exposure times (in seconds) as a second argument; it produces a physically meaningful radiance map.

Once you have an `Image<Float32C3>`, pass it to `ToneMap` to produce a displayable 8-bit `Image<BGR>`. Four algorithms are available: `Reinhard` (perceptually natural, default), `Linear` (global scale), `Drago` (good contrast in dark regions), and `Mantiuk` (perceptual contrast preservation). The `gamma` setter applies post-tone-mapping gamma correction; set it to 1.0 for no correction.

```cpp
// HDR pipeline: MergeHDR → Image<Float32C3> → ToneMap → Image<BGR>
// Mertens: exposure fusion (no exposure times required)
Image<Float32C3> hdr_mertens = MergeHDR{}
    .method(MergeHDR::Method::Mertens)  // Mertens (default) or Debevec
    (images);                            // std::vector<Image<BGR>>

// Debevec: true HDR from bracketed exposures (exposure times required)
Image<Float32C3> hdr_debevec = MergeHDR{}
    .method(MergeHDR::Method::Debevec)
    (images, exposure_times);            // second arg: std::vector<float>

// ToneMap: maps Float32C3 back to displayable BGR
Image<BGR> tm_reinhard = ToneMap{}
    .gamma(1.f)                               // gamma correction (default: 1)
    .algorithm(ToneMap::Algorithm::Reinhard)  // Reinhard, Linear, Drago, Mantiuk
    (hdr_mertens);
```

## Panorama Stitching (Stitch)

`Stitch` wraps OpenCV's `cv::Stitcher` to align and blend two or more overlapping images into a single panorama. The result is returned as a `StitchResult` struct with an `ok` flag and the final `panorama` image. If stitching fails — typically because images share too little overlap or have too few detectable features — `ok` is `false` and the panorama field holds a 1×1 placeholder. The `Scans` mode is intended for flat document or flatbed scanner compositing where projective distortion is not expected.

```cpp
// Stitch — panorama stitching from overlapping images
StitchResult sr = Stitch{}
    .mode(Stitch::Mode::Panorama)   // Panorama (default) or Scans
    (images);                        // std::vector<Image<BGR>>

if (sr.ok)
    cv::imshow("Panorama", sr.panorama.mat());
else
    std::cerr << "Stitching failed — too little overlap?\n";
```

## Summary Table

| Op | Purpose | Key Setters | Returns |
|---|---|---|---|
| EdgePreservingFilter | Edge-aware smoothing | sigma_s, sigma_r, filter | Image\<BGR\> |
| DetailEnhance | Sharpen fine detail | sigma_s, sigma_r | Image\<BGR\> |
| Stylize | Oil-painting effect | sigma_s, sigma_r | Image\<BGR\> |
| PencilSketch | Pencil sketch | sigma_s, sigma_r, shade_factor | PencilSketchResult{gray, color} |
| SeamlessClone | Poisson compositing | mode | Image\<BGR\> |
| NLMeansDenoisingMulti | Temporal denoising | h, temporal_window_size | Image\<BGR\> |
| MergeHDR | Exposure fusion / HDR | method | Image\<Float32C3\> |
| ToneMap | HDR → LDR tone mapping | gamma, algorithm | Image\<BGR\> |
| Stitch | Panorama stitching | mode | StitchResult{ok, panorama} |

## Next Steps
- [Image Quality Metrics](image-quality.md)
- [Perceptual Hashing](perceptual-hashing.md)
- [Camera Calibration](camera-calibration.md)
