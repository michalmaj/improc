# Object Detectors

improc++ provides two distinct families of detectors under `improc::core`. The first family — **feature detectors** (ORB, SIFT, AKAZE, covered in [Feature Detection and Matching](feature-detection.md)) — finds keypoints and computes descriptors suitable for cross-image matching, homography estimation, and structure-from-motion. The second family — **structure and region detectors** — locates specific image structures without producing matchable descriptors. This tutorial focuses entirely on the second family.

Structure detectors are the right choice when you need to find *what is in the image* rather than *where one image aligns with another*. FAST gives you a flood of corner candidates in microseconds, making it ideal for tracking initialisers. Blob detection finds round objects by area and circularity. MSER extracts stable high-contrast regions, commonly used for text detection and logo segmentation. LSD (Line Segment Detector) recovers line primitives that are invaluable for lane detection, table extraction, and grid analysis. QR and barcode detectors go all the way to decoded content, while face detection yields bounding boxes and five-point landmarks ready for downstream recognition.

Choosing between these detectors comes down to two questions: *what is the structure you expect to find*, and *do you need decoded content or geometric primitives*? Speed matters too — FAST is essentially free, LSD is moderate, and QR/barcode decode paths are heavier but still sub-millisecond on modern hardware for typical frame sizes.

All detectors follow the same pipeline idiom: `result = image | DetectXxx{}.setter(value);`. The type returned depends on the detector — `KeypointSet`, `MSERResult`, `LineSet`, `QRResult`, or `BarcodeResult`. All result types expose `.size()` and `.empty()`.

## Prerequisites

- Completed [Feature Detection and Matching](feature-detection.md)
- `improc/core/pipeline.hpp` (umbrella header — includes all core ops)

## FAST Corner Detection

FAST (Features from Accelerated Segment Test) tests whether a circular arc of pixels around a candidate is all brighter or all darker than the centre by a given threshold. Because the test short-circuits on the first non-satisfying pixel, FAST is by far the cheapest corner detector in OpenCV — often 10–100× faster than SIFT or even ORB detection. It is commonly used to seed sparse optical-flow trackers or as a first pass before a slower descriptor stage. The `threshold` parameter controls sensitivity: lower values find more, weaker corners; higher values keep only strong, high-contrast ones. Non-maxima suppression retains only local response maxima, reducing duplicate detections on edges.

```cpp
#include "improc/core/pipeline.hpp"
#include "improc/io/image_io.hpp"
using namespace improc::core;

Image<Gray> gray = /* gray image */;

KeypointSet kps = gray | DetectFAST{}
    .threshold(10)               // contrast threshold (default: 10)
    .non_max_suppression(true);  // suppress adjacent responses (default: true)

std::cout << "FAST: " << kps.size() << " corners\n";
// kps.keypoints is std::vector<cv::KeyPoint>
// High threshold → fewer, stronger corners; low threshold → more detections
```

`KeypointSet` is shared with the feature-detector family. You can draw the corners with `cv::drawKeypoints` or pass the set to a `Describe*` op for a downstream matching stage.

## Blob Detection

A *blob* is a connected region that stands out from its surroundings by colour, area, circularity, convexity, or inertia. `DetectBlob` wraps `cv::SimpleBlobDetector`, which first thresholds the image at multiple levels, groups connected components at each level, and merges them across thresholds to find stable blobs. The result is a `KeypointSet` where each keypoint's `.size` field encodes the blob's diameter and `.pt` its centre.

```cpp
cv::SimpleBlobDetector::Params params;
params.minArea         = 100;    // minimum blob area in pixels (default: 25)
params.maxArea         = 5000;   // maximum blob area in pixels (default: 5000)
params.minCircularity  = 0.8f;  // 1.0 = perfect circle (default: 0.8)
params.minConvexity    = 0.87f; // 1.0 = convex hull (default: 0.87)
params.minInertiaRatio = 0.01f; // 1.0 = circle, 0.0 = line (default: 0.01)

KeypointSet blobs = gray | DetectBlob{}.params(params);
std::cout << "Blobs: " << blobs.size() << "\n";
// keypoint.size encodes the blob diameter; .pt the centre
```

To detect *light* blobs on a dark background, set `params.blobColor = 255`; for dark blobs on a light background, set `params.blobColor = 0`. Disable individual filter groups (`filterByArea`, `filterByCircularity`, etc.) that are not relevant to your use case to improve recall.

## MSER Regions

Maximally Stable Extremal Regions (MSER) sweeps the image through a range of intensity thresholds and records connected components that remain stable — i.e., their area changes slowly across threshold steps. Stability is controlled by the `delta` parameter: a region survives if its relative area change over `delta` thresholds is below `max_variation`. MSER is particularly effective for detecting text, logos, and other high-contrast compact regions because such structures tend to be stable across a wide threshold range. The result is an `MSERResult` with two parallel vectors: `.regions` (pixel lists) and `.bboxes` (axis-aligned bounding boxes).

```cpp
MSERResult mser = gray | DetectMSER{}
    .delta(5)        // stability threshold — larger = fewer, more stable regions (default: 5)
    .min_area(60)    // minimum region pixel area (default: 60)
    .max_area(14400); // maximum region pixel area (default: 14400)

std::cout << "MSER regions: " << mser.size() << "\n";
// mser.regions[i]: vector<cv::Point> — all pixels of the region
// mser.bboxes[i]:  cv::Rect bounding box

// Draw bounding boxes
cv::Mat vis = /* BGR image */ .mat().clone();
for (const auto& bbox : mser.bboxes)
    cv::rectangle(vis, bbox, {0, 255, 0}, 1);
```

If you only need approximate shapes, work with `.bboxes`. If you need exact pixel membership (e.g., to build a mask), iterate `.regions[i]` directly.

## Line Segment Detection (LSD)

LSD (Line Segment Detector) extracts straight line segments from an image without tuning and without needing any ground-truth training. It works by estimating the local gradient orientation at each pixel, then grouping pixels whose orientations are consistent into line-support regions, and finally fitting segments to those regions. The `scale` parameter downsamples the image before detection (values below 1.0 speed things up at the cost of resolving shorter segments), and `sigma_scale` controls the pre-smoothing relative to the image scale. LSD excels at finding lane markings, table grid lines, building edges, and document layout boundaries. Each detected segment is returned as a `cv::Vec4f` of `(x1, y1, x2, y2)` endpoints.

```cpp
LineSet ls = gray | DetectLines{}
    .scale(0.8)        // image scale for detection (default: 0.8)
    .sigma_scale(0.6); // Gaussian blur scale (default: 0.6)

std::cout << "Lines: " << ls.size() << "\n";
// ls.lines[i]: cv::Vec4f — (x1, y1, x2, y2) endpoints

cv::Mat vis = /* BGR */ .mat().clone();
for (const auto& l : ls.lines)
    cv::line(vis, {(int)l[0], (int)l[1]}, {(int)l[2], (int)l[3]},
             {0, 0, 255}, 1);
```

LSD returns many short segments along textured regions. A common post-processing step is to merge nearly-collinear segments or filter by minimum length using the endpoint distance: `cv::norm(cv::Point2f(l[0],l[1]) - cv::Point2f(l[2],l[3]))`.

## QR Code Detection

`DetectQR` wraps `cv::QRCodeDetectorAruco` (available from OpenCV 4.8 onward). It detects one or more QR codes in an `Image<BGR>`, decodes their content, and returns a `QRResult`. Each detected code occupies one slot in the parallel vectors `.decoded` (the string payload) and `.points` (a `CV_32FC2` 4×1 `cv::Mat` holding the four polygon corners). The detector is robust to moderate perspective distortion and partial occlusion, and it requires no model files.

```cpp
Image<BGR> img = /* load image with QR code(s) */;

QRResult qr = img | DetectQR{};
std::cout << "QR codes detected: " << qr.size() << "\n";
for (std::size_t i = 0; i < qr.size(); ++i) {
    std::cout << "  [" << i << "] decoded: \"" << qr.decoded[i] << "\"\n";
    // qr.points[i]: 4-corner polygon (CV_32FC2, 4×1 mat)
}
```

To draw the detected polygon corners, iterate `qr.points[i]` with `at<cv::Point2f>(k, 0)` for `k` in `[0, 3]` and draw with `cv::polylines`. QR codes embedded in padded white canvases (as generated by `cv::QRCodeEncoder`) detect most reliably; leave at least 4 module widths of quiet zone around the code.

## Barcode Detection

`DetectBarcode` decodes 1-D and 2-D barcodes — ISBN, EAN-8, EAN-13, UPC-A, UPC-E, Code-39, Code-128, and more — using OpenCV's built-in barcode module. No model file or licence is required. The result is a `BarcodeResult` with three parallel vectors: `.decoded` (the string content), `.types` (the format name, e.g. `"EAN_13"`), and `.bboxes` (a `cv::RotatedRect` giving the oriented bounding box in the original image). The detector handles moderate in-plane rotation and mild perspective distortion.

```cpp
BarcodeResult bc = img | DetectBarcode{};
std::cout << "Barcodes detected: " << bc.size() << "\n";
for (std::size_t i = 0; i < bc.size(); ++i) {
    std::cout << "  [" << i << "] type=" << bc.types[i]
              << "  decoded=\"" << bc.decoded[i] << "\"\n";
    // bc.bboxes[i]: cv::RotatedRect
}
```

For best results, ensure the barcode covers at least 10% of the image width and that the image is sharply focused. If you are reading barcodes from video frames, consider running `DetectBarcode` only every N frames to limit CPU usage.

## Face Detection (model-gated)

`DetectFaceYN` and `RecognizeFace` require model files downloaded separately from the OpenCV Model Zoo. `DetectFaceYN` takes a `yunet.onnx` model path and returns a `std::vector<FaceDetection>`, each holding a `cv::Rect2f` bounding box, a confidence score, and five facial landmarks (right eye, left eye, nose tip, right mouth corner, left mouth corner). `RecognizeFace` takes an `sface.onnx` model path and produces a `FaceEmbedding` (a typed wrapper around a `(1, 128) CV_32F` descriptor) per aligned face crop; use the static `RecognizeFace::match(emb_a, emb_b)` function to compute cosine similarity — a value above 0.363 corresponds to same-identity at FAR = 1 × 10⁻⁵.

```cpp
// Requires model files downloaded from the OpenCV Model Zoo
// DetectFaceYN: returns vector<FaceDetection> {bbox, confidence, landmarks}
// RecognizeFace: returns FaceEmbedding; use RecognizeFace::match(emb_a, emb_b)
//                for cosine similarity (>0.363 = same person at FAR=1e-5)
```

Download links and a complete face recognition walkthrough are in [ML Inference](ml-inference.md).

## Choosing the Right Detector

| Detector | Speed | Descriptor? | Best for |
|---|---|---|---|
| `DetectFAST` | Very fast | No | Tracking seed points, real-time corner finding |
| `DetectBlob` | Fast | No | Round objects (cells, coins, bubbles), dot grids |
| `DetectMSER` | Moderate | No | Text regions, logos, high-contrast compact shapes |
| `DetectLines` | Moderate | No | Lane markings, table/grid edges, building outlines |
| `DetectQR` | Moderate | No (decoded content) | QR code scanning, URL and config retrieval |
| `DetectBarcode` | Moderate | No (decoded content) | 1-D barcode scanning (EAN, UPC, Code-128) |
| `DetectFaceYN` | Moderate (model required) | No | Face localisation and landmark detection |

When you need both detection and subsequent *image-to-image matching*, prefer the feature detectors (ORB, SIFT, AKAZE) from [Feature Detection and Matching](feature-detection.md).

## Next Steps

- [Camera Calibration](camera-calibration.md)
- [Motion Analysis](motion-analysis.md)
- [Photo and Creative Effects](photo-creative.md)
