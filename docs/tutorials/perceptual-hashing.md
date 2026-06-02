# Perceptual Hashing

A perceptual hash is a compact fingerprint of an image's visual content rather than its raw bytes. Two images that look similar to a human observer will produce hashes with a small distance between them, and two images that look completely different will produce hashes that are far apart. This behaviour is the opposite of cryptographic hashing — a single changed byte in a JPEG produces a wildly different SHA-256 but barely perturbs a perceptual hash, because the hash is derived from perceptual structure (frequency content, colour statistics, spatial layout) rather than exact byte values.

The practical applications of perceptual hashing are broad. Near-duplicate detection identifies re-uploaded, recompressed, or slightly cropped versions of the same photograph in a content platform without storing or comparing full pixel data. Content deduplication in large image archives eliminates redundant copies even when file formats, resolutions, or colour profiles have changed. Visual search pipelines use hashes as a fast pre-filter: rather than comparing a query against millions of full images, a hash lookup narrows the candidate set to a handful of similar-looking images for detailed verification. The hash distance also provides a useful similarity score for ranking results.

`improc++` provides six perceptual hash algorithms, each tuned to a different trade-off between speed, robustness, and discrimination power. All six take an `Image<BGR>` and return an `ImageHash` (a typed wrapper around a `cv::Mat`), and each exposes a static `distance()` method that computes the canonical distance metric for that algorithm. Access the raw matrix via `.value` if needed. All are implemented using standard OpenCV — no contrib module is required.

## Prerequisites
- Completed [Building a Pipeline](building-a-pipeline.md)
- `improc/core/pipeline.hpp`

## AverageHash

`AverageHash` is the simplest and fastest perceptual hash. It rescales the image to 8×8 pixels, converts to grayscale, computes the mean pixel value, and encodes each pixel as a 1-bit flag indicating whether it is above or below the mean. The resulting 64-bit hash is stored as a 1×8 `CV_8U` matrix. Distance is Hamming — the count of differing bits. `AverageHash` is highly sensitive to global brightness changes and colour shifts, making it best suited for exact-duplicate detection where exposure and colour balance are known to be consistent.

```cpp
#include "improc/core/pipeline.hpp"
using namespace improc::core;

Image<BGR> img = /* load image */;

// AverageHash — 8×8 resize → mean threshold → 64-bit hash (1×8 CV_8U)
// Fastest; sensitive to colour/brightness changes
ImageHash ah = AverageHash{}(img);
double    ah_dist = AverageHash::distance(ah, ah2);  // Hamming distance
```

## PHash (DCT Hash)

`PHash` applies a Discrete Cosine Transform to a 32×32 grayscale reduction of the image, then encodes the sign structure of the 8×8 top-left block of low-frequency DCT coefficients as a 63-bit hash (stored as 1×8 `CV_8U`). Because the DCT concentrates the perceptual content of the image in low-frequency components, `PHash` is significantly more robust than `AverageHash` to brightness changes, minor colour shifts, and mild compression artefacts. It is the recommended default for general-purpose near-duplicate detection.

```cpp
// PHash (DCT hash) — 32×32 → DCT → 8×8 top-left → 63-bit hash (1×8 CV_8U)
// More robust than AverageHash; best general-purpose choice
ImageHash ph = PHash{}(img);
double    ph_dist = PHash::distance(ph, ph2);         // Hamming
```

## MarrHildrethHash

`MarrHildrethHash` computes a Laplacian-of-Gaussian (LoG) response on a 24×24 grayscale reduction and encodes the zero-crossing sign map as a 576-bit hash (stored as 1×72 `CV_8U`). The LoG operator highlights edges and blobs at a scale determined by its Gaussian sigma, and the sign of the response at each pixel captures the spatial structure of those features robustly. The large hash size provides fine discrimination, and the LoG's inherent noise suppression makes this algorithm well-suited to moderately distorted, noisy, or watermarked images.

```cpp
// MarrHildrethHash — LoG sign bits on 24×24 → 576-bit hash (1×72 CV_8U)
// Robust to noise and moderate geometric distortion
ImageHash mh = MarrHildrethHash{}(img);
double    mh_dist = MarrHildrethHash::distance(mh, mh2); // Hamming
```

## RadialVarianceHash

`RadialVarianceHash` samples the variance of 40 radial projections through the image centre, producing a 40-element vector of `double` values stored as a 1×40 `CV_64F` matrix. Because the projections are computed rotationally, the hash is naturally rotation-aware — rotating the image changes the starting angle of the projections but preserves their variance distribution, keeping the distance low. This makes `RadialVarianceHash` the right choice when the image set includes photographs of the same scene taken from different orientations. Distance is L2 (Euclidean) rather than Hamming.

```cpp
// RadialVarianceHash — 40 radial variance samples → 40-float hash (1×40 CV_64F)
// Rotation-aware; use when images may be rotated
ImageHash rv = RadialVarianceHash{}(img);
double    rv_dist = RadialVarianceHash::distance(rv, rv2); // L2
```

## ColorMomentHash

`ColorMomentHash` converts the image to YCrCb colour space and computes statistical colour moments (mean, variance, skewness) per channel, producing a 42-element descriptor stored as a 1×42 `CV_64F` matrix. It captures the global colour distribution of an image in a compact form, making it useful for matching images that share the same colour palette even when spatial layout differs. It is ineffective for finding structurally identical images with different colour grading. Distance is L2.

```cpp
// ColorMomentHash — 42 YCrCb colour statistics → 42-float hash (1×42 CV_64F)
// Captures global colour distribution
ImageHash cm = ColorMomentHash{}(img);
double    cm_dist = ColorMomentHash::distance(cm, cm2);    // L2
```

## BlockMeanHash

`BlockMeanHash` rescales the image to 256×256, divides it into a grid of non-overlapping blocks, computes the mean of each block, and encodes whether each block mean is above or below the global mean as a single bit, producing a 256-bit hash stored as a 1×32 `CV_8U` matrix. Unlike `AverageHash`, which collapses the image to a single 8×8 grid, `BlockMeanHash` retains more spatial information about where bright and dark regions occur. It is particularly effective for large images where spatial layout carries meaningful discriminative content. Distance is Hamming.

```cpp
// BlockMeanHash — block mean threshold on 256×256 → 256-bit hash (1×32 CV_8U)
// Layout-sensitive; suitable for large images where spatial structure matters
ImageHash bm = BlockMeanHash{}(img);
double    bm_dist = BlockMeanHash::distance(bm, bm2);      // Hamming
```

## Practical Distance Thresholds

The right similarity threshold is always dataset-dependent and should be tuned empirically. The values below are reasonable starting points for near-duplicate detection on photographic images with minor recompression or minor brightness adjustments:

- **PHash Hamming ≤ 10** → likely near-duplicate
- **AverageHash Hamming ≤ 10** → likely near-duplicate
- **MarrHildrethHash Hamming ≤ 60** → likely similar
- **RadialVarianceHash L2 ≤ 0.3** → likely similar
- **ColorMomentHash L2 ≤ 3.0** → similar colour distribution
- **BlockMeanHash Hamming ≤ 25** → likely similar layout

These thresholds will be too strict for highly compressed inputs or images subjected to geometric transformation, and may be too lenient for datasets with many visually similar but semantically distinct images. Always validate on a representative sample before deploying a pipeline.

## Algorithm Comparison

| Algorithm | Hash size | Distance | Speed | Strength |
|---|---|---|---|---|
| AverageHash | 64 bits | Hamming | Fastest | Brightness-sensitive |
| PHash | 63 bits | Hamming | Fast | Best general-purpose |
| MarrHildrethHash | 576 bits | Hamming | Medium | Noise/distortion robust |
| RadialVarianceHash | 40 floats | L2 | Medium | Rotation-aware |
| ColorMomentHash | 42 floats | L2 | Fast | Colour-sensitive |
| BlockMeanHash | 256 bits | Hamming | Fast | Spatial layout |

## Next Steps
- [Image Quality Metrics](image-quality.md)
- [Photo and Creative Effects](photo-creative.md)
