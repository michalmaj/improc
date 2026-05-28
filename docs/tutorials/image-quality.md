# Image Quality Metrics

Image quality metrics quantify the fidelity of a processed or compressed image relative to a pristine reference. Unlike perceptual hashing, which asks "are these two images the same scene?", quality metrics ask "how much has the pixel-level content degraded?". They are indispensable in full-reference evaluation workflows where a ground-truth image is available alongside a candidate produced by a compression codec, a generative model, a filter, or a super-resolution network.

`improc++` provides four classic full-reference metrics — PSNR, SSIM, GMSD, and MSE — each accessible via a zero-configuration functor in `improc::core`. The choice of metric depends on the application: codec comparison typically favours PSNR because of its historical precedence, perceptual quality studies prefer SSIM, gradient-sensitive tasks (blurring detection, sharpness evaluation) benefit from GMSD, and simple numerical baselines use MSE. In practice, reporting two or more complementary metrics gives a more complete picture than any single value.

All four ops work with both colour (`Image<BGR>`) and grayscale (`Image<Gray>`) inputs. For BGR images, the metric is computed per channel and the results are averaged to produce a single scalar. Both the reference and the comparison image must have the same width and height; a mismatch throws `improc::ParameterError`. The reference and comparison images are the two arguments to `operator()`.

## Prerequisites
- Completed [Building a Pipeline](building-a-pipeline.md)
- `improc/core/pipeline.hpp`

## Using the Quality Ops

All four ops are plain functors: construct one with default `{}`, then call it with a reference image and a comparison image. Each returns a `double`. There are no setters or state — the entire interface is a single call. The ops do not modify their arguments (both are taken by const reference), so the same reference image can be passed to multiple metric calls without copying.

```cpp
#include "improc/core/pipeline.hpp"
#include "improc/io/image_io.hpp"
using namespace improc::core;
using namespace improc::io;

auto ref = imread<BGR>("reference.png");
auto cmp = imread<BGR>("compressed.png");
if (!ref || !cmp) return 1;
// Both must be the same size — throws ParameterError otherwise

// PSNR: higher = better; returns INFINITY when images are identical (MSE == 0)
double psnr = PSNR{}(*ref, *cmp);
std::cout << "PSNR: " << psnr << " dB";
// >40 dB = excellent; 30–40 dB = good; <30 dB = noticeable artifacts

// SSIM: structural similarity; range [-1, 1]; 1.0 = identical
double ssim = SSIM{}(*ref, *cmp);
std::cout << "SSIM: " << ssim;

// GMSD: gradient magnitude deviation; lower = better; 0.0 = identical
double gmsd = GMSD{}(*ref, *cmp);
std::cout << "GMSD: " << gmsd;

// MSE: mean squared error; lower = better; 0.0 = identical
double mse = MSE{}(*ref, *cmp);
std::cout << "MSE: " << mse;

// Gray overloads work identically
auto ref_g = imread<Gray>("reference_gray.png");
auto cmp_g = imread<Gray>("compressed_gray.png");
double psnr_g = PSNR{}(*ref_g, *cmp_g);
```

## PSNR Formula and Interpretation

PSNR (Peak Signal-to-Noise Ratio) is defined as **10·log₁₀(255²/MSE)**, where 255 is the maximum pixel value for 8-bit images and MSE is the mean squared error. When the two images are identical, MSE is zero and PSNR is positive infinity; `improc++` returns `std::numeric_limits<double>::infinity()` in this case. Thresholds are application-dependent but a useful rule of thumb is: above 40 dB is excellent quality (differences are imperceptible), 30–40 dB is good (minor artefacts visible under magnification), and below 30 dB produces noticeable degradation. PSNR is inexpensive to compute and easy to interpret logarithmically, which makes it the dominant metric in codec standardisation work, but it is known to correlate poorly with human perception for non-Gaussian distortions such as blurring or structural changes.

## SSIM Details

SSIM (Structural Similarity Index) evaluates image quality on three axes — luminance, contrast, and structure — using overlapping 11×11 Gaussian windows across the image, then combines those per-window scores into a single global value. For BGR inputs the metric is computed independently per channel and the channel averages are combined. The result lies in [−1, 1], where 1.0 means identical images. Unlike PSNR, SSIM is sensitive to correlations between pixel neighbourhoods, making it more robust to small spatial shifts and better aligned with human judgement of structural distortions such as ringing and blocking artefacts. The implementation uses the standard constants C1 = 6.5025 and C2 = 58.5225.

## GMSD Details

GMSD (Gradient Magnitude Similarity Deviation) computes the per-pixel ratio of gradient magnitudes between the reference and the comparison image and then measures the standard deviation of that similarity map. The gradient is computed on a grayscale conversion of the input using a Prewitt kernel. Low standard deviation means the distortion is spatially uniform; high deviation indicates localised damage. The metric returns 0.0 for identical images and increases with degradation — lower is better. GMSD is faster than SSIM (single-pass gradient + scalar statistics), correlates well with perceptual quality for blurring and compression, and is particularly useful for detecting spatially heterogeneous distortions.

## MSE Details

MSE (Mean Squared Error) is the arithmetic mean of the squared per-pixel differences between the reference and the comparison. For BGR images the differences across all three channels are included in the average. It is the baseline metric upon which PSNR is built and is the cheapest to compute. MSE is maximally sensitive to global brightness shifts — adding a constant offset to every pixel gives a non-zero MSE even if the structural content is unchanged — which is one of the reasons it correlates poorly with perceived quality in many scenarios. It is most useful as a fast sanity-check or as a loss term in optimisation, not as a perceptual quality judge.

## Comparison Table

| Metric | Formula | Range | Identical | Better | Cost |
|---|---|---|---|---|---|
| PSNR | 10·log₁₀(255²/MSE) | (0, +∞] dB | +∞ | Higher | Low |
| SSIM | Structural similarity | [-1, 1] | 1.0 | Higher | Medium |
| GMSD | Gradient magnitude deviation | [0, +∞) | 0.0 | Lower | Medium |
| MSE | Mean squared error | [0, +∞) | 0.0 | Lower | Low |

## Next Steps
- [Perceptual Hashing](perceptual-hashing.md)
- [Photo and Creative Effects](photo-creative.md)
