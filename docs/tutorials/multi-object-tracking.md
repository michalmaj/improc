# Multi-Object Tracking

This tutorial shows how to track detections across frames using improc++'s three tracker implementations: `IouTracker`, `SortTracker`, and `ByteTracker`. It also covers visualizing tracks with `DrawTracks` and measuring quality with `TrackingEval`.

## Prerequisites

- Completed [ML Inference](ml-inference.md)
- A video source or a sequence of frames with a detector

## The Tracker API

All three trackers satisfy the `TrackerType` concept:

```cpp
// concept TrackerType requires:
tracker.update(std::vector<Detection> dets)  // → std::vector<Track>
tracker.reset()                              // → void
```

`Track` fields:
| Field | Type | Description |
|---|---|---|
| `id` | `int` | Unique track ID (stable across frames) |
| `bbox` | `BBox` | Current bounding box |
| `confidence` | `float` | Detection confidence of the most recent match |
| `age` | `int` | Number of frames this track has existed |
| `is_confirmed` | `bool` | `true` once the track has passed `min_hits` |

## IouTracker — Simple IoU Matching

`IouTracker` matches detections to existing tracks by highest IoU. Tracks that go unmatched for `max_age` consecutive frames are dropped. No Kalman filter — fast and easy to understand.

```cpp
#include "improc/ml/tracking/tracking.hpp"
using namespace improc::ml;

IouTracker tracker{};
tracker.min_iou(0.3f)   // minimum IoU to match (default: 0.3)
       .max_age(1);     // frames a track survives without a match (default: 1)

std::vector<Track> tracks = tracker.update(detections);
```

Good for: stationary cameras, slow-moving objects, fast prototyping.

## SortTracker — Kalman Filter + IoU

`SortTracker` uses a Kalman filter to predict each track's next position, then matches predictions to new detections by IoU. Tracks are only reported once they've been confirmed by `min_hits` consecutive matches.

```cpp
#include "improc/ml/tracking/tracking.hpp"
using namespace improc::ml;

SortTracker tracker{};
tracker.max_age(3)           // frames without match before track dies (default: 3)
       .min_hits(3)          // consecutive matches before track is confirmed (default: 3)
       .iou_threshold(0.3f); // minimum IoU for assignment (default: 0.3)

std::vector<Track> tracks = tracker.update(detections);
```

Good for: general-purpose MOT with moderate occlusion.

## ByteTracker — Two-Stage Association

`ByteTracker` extends SORT by performing a two-stage match: high-confidence detections are matched first; remaining unmatched tracks get a second chance to match low-confidence detections. This recovers objects that briefly drop below the detector's threshold.

```cpp
#include "improc/ml/tracking/tracking.hpp"
using namespace improc::ml;

ByteTracker tracker{};
tracker.max_age(3)
       .min_hits(3)
       .high_conf_threshold(0.6f)   // detections above this go to stage 1 (default: 0.6)
       .low_conf_threshold(0.1f);   // detections above this go to stage 2 (default: 0.1)

std::vector<Track> tracks = tracker.update(detections);
```

Good for: crowded scenes, high-speed objects, or whenever detections are noisy.

## Choosing a Tracker

| Tracker | Kalman filter | Two-stage match | When to use |
|---|---|---|---|
| `IouTracker` | No | No | Quick experiments, sparse scenes |
| `SortTracker` | Yes | No | General-purpose MOT |
| `ByteTracker` | Yes | Yes | Noisy detectors, crowded scenes |

## Video Processing Loop

```cpp
#include "improc/ml/ml.hpp"
#include "improc/ml/tracking/tracking.hpp"
#include "improc/visualization/visualization.hpp"
#include "improc/io/image_io.hpp"
#include "improc/threading/threading.hpp"
using namespace improc::ml;
using namespace improc::visualization;
using namespace improc::io;

int main() {
    std::vector<std::string> labels = {"person", "car", "bicycle"};

    DnnDetector det{"yolov8n.onnx"};
    det.input_size(640, 640)
       .confidence_threshold(0.4f)
       .nms_threshold(0.4f)
       .labels(labels);

    ByteTracker tracker{};
    tracker.high_conf_threshold(0.6f).low_conf_threshold(0.15f);

    CameraCapture cam{0};  // or path to video file

    while (true) {
        auto frame = cam.next();
        if (!frame) break;

        std::vector<Detection> dets   = det(*frame);
        std::vector<Track>     tracks = tracker.update(dets);

        // Draw tracks: shows box + ID for confirmed tracks
        *frame | DrawTracks{tracks}.thickness(2) | Show{"Tracking"};

        if (cv::waitKey(1) == 'q') break;
    }
}
```

## Visualizing Tracks with `DrawTracks`

`DrawTracks` is a pipeline op that draws bounding boxes with persistent colour-coded IDs. Only confirmed tracks (`is_confirmed == true`) are drawn by default.

```cpp
#include "improc/visualization/visualization.hpp"
using namespace improc::visualization;

// Draw over a copy
auto annotated = *frame | DrawTracks{tracks}.thickness(2);

// Draw all tracks (including unconfirmed)
auto annotated2 = *frame | DrawTracks{tracks}.confirmed_only(false);
```

Each track ID gets a consistent colour across frames so you can visually follow individual objects.

## Evaluating Tracking Quality with `TrackingEval`

```cpp
#include "improc/ml/tracking/tracking_eval.hpp"
using namespace improc::ml;

TrackingEval eval{};
eval.iou_threshold(0.5f);

for (auto& frame : annotated_video) {
    std::vector<Track>     tracks = tracker.update(detector(frame.image));
    std::vector<TrackGT>   gts    = frame.ground_truth;  // persistent IDs per frame
    eval.update(tracks, gts);
}

auto m = eval.compute();
std::cout << "MOTA: " << m.MOTA  << "  MOTP: " << m.MOTP
          << "  IDF1: " << m.IDF1 << "\n";
std::cout << "FP=" << m.FP << "  FN=" << m.FN << "  IDSW=" << m.IDSW << "\n";
```

See [Evaluation Metrics](evaluation-metrics.md) for a full description of each metric.

## Resetting a Tracker Between Sequences

Call `reset()` when processing a new video clip to clear all internal state:

```cpp
tracker.reset();
```

## Template-Generic Code with `TrackerType`

All three trackers satisfy the `TrackerType` concept, so you can write tracker-agnostic code:

```cpp
#include "improc/ml/tracking/track.hpp"
using namespace improc::ml;

template<TrackerType T>
void run_tracking(T& tracker, const std::vector<std::vector<Detection>>& frames) {
    for (const auto& dets : frames) {
        auto tracks = tracker.update(dets);
        // ... process tracks ...
    }
}

// Call with any tracker:
IouTracker iou_tr{};   run_tracking(iou_tr, frames);
SortTracker sort_tr{}; run_tracking(sort_tr, frames);
ByteTracker byte_tr{}; run_tracking(byte_tr, frames);
```

## Next Steps

- [Evaluation Metrics](evaluation-metrics.md) — MOTA, MOTP, IDF1 explained in detail
- [ML Inference](ml-inference.md) — the detector that feeds the tracker
- [ML Visualization Charts](ml-visualization-charts.md) — plot per-class metrics from a tracking run
