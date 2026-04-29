# Real-Time Camera Processing

This tutorial shows how to capture frames from a camera, process them in a thread pool, and display or record the results.

## Components

| Class | Responsibility |
|---|---|
| `CameraCapture` | Asynchronous frame capture in a background thread |
| `ThreadPool` | Fixed-size worker pool; `submit()` returns `std::future<T>` |
| `FramePipeline<Result>` | Ties capture + pool together; `tryPop()` returns `std::optional<Result>` |
| `VideoWriter` | RAII video recording; pipeline-composable via `operator\|` |
| `Show` | Passthrough display op; returns the image unchanged after showing it |

## Minimal Live Preview

The simplest case — capture and display on the main thread:

```cpp
#include "improc/core/pipeline.hpp"
#include "improc/io/camera_capture.hpp"
#include "improc/visualization/visualization.hpp"

using namespace improc::core;
using namespace improc::io;
using namespace improc::visualization;

int main() {
    CameraCapture cam(0);   // device index 0

    while (true) {
        auto frame = cam.getFrame();  // std::expected<cv::Mat, Error>
        if (!frame) continue;

        Image<BGR> img(*frame);
        img | Show{"Live"}.wait_ms(1);  // display, wait 1 ms

        if (cv::waitKey(1) == 27) break;  // ESC to quit
    }
}
```

## Capture + Process + Record

Process frames in the main thread and record them to a file simultaneously. `VideoWriter` is pipeline-composable — `img | Show{...} | writer` displays and records in one expression:

```cpp
#include "improc/core/pipeline.hpp"
#include "improc/io/camera_capture.hpp"
#include "improc/io/video_writer.hpp"
#include "improc/visualization/visualization.hpp"

using namespace improc::core;
using namespace improc::io;
using namespace improc::visualization;

int main() {
    CameraCapture cam(0);
    VideoWriter writer{"output.mp4"};
    writer.fps(30);

    while (true) {
        auto frame = cam.getFrame();
        if (!frame) continue;

        Image<BGR> img(*frame);

        // Apply processing, display, and record in one pipeline
        img
            | GaussianBlur{}.kernel_size(3)
            | Brightness{}.delta(10.0)
            | Show{"Preview"}.wait_ms(1)
            | writer;

        if (cv::waitKey(1) == 27) break;
    }
    // VideoWriter destructor finalises the file
}
```

## Threaded Processing with FramePipeline

For computationally heavy per-frame work (inference, augmentation, multi-step pipelines), offload processing to a `ThreadPool` so the capture loop never blocks:

```cpp
#include <iostream>
#include <optional>
#include "improc/core/pipeline.hpp"
#include "improc/io/camera_capture.hpp"
#include "improc/threading/thread_pool.hpp"
#include "improc/threading/frame_pipeline.hpp"
#include "improc/visualization/visualization.hpp"

using namespace improc::core;
using namespace improc::io;
using namespace improc::threading;
using namespace improc::visualization;

// The result type your pipeline produces per frame
struct FrameResult {
    Image<BGR> processed;
};

int main() {
    CameraCapture cam(0);
    ThreadPool pool(4);  // 4 worker threads

    // FramePipeline holds *references* — cam and pool must outlive it
    FramePipeline<FrameResult> pipeline(cam, pool);

    pipeline.start([](const cv::Mat& raw) -> FrameResult {
        // This lambda runs on a worker thread
        Image<BGR> img(raw);
        return {
            img
                | Resize{}.width(640).height(360)
                | GaussianBlur{}.kernel_size(5)
                | CLAHE{}.clip_limit(2.0)
        };
    });

    while (true) {
        // tryPop() is non-blocking — returns std::nullopt if no result ready
        if (auto result = pipeline.tryPop()) {
            result->processed | Show{"Processed"}.wait_ms(1);
        }

        if (cv::waitKey(1) == 27) break;
    }

    pipeline.stop();
}
```

`FramePipeline` submits one task per frame to the pool. The pool processes frames concurrently; results are queued in arrival order and retrieved with `tryPop()`.

## ThreadPool Directly

For custom parallelism patterns, use `ThreadPool` directly:

```cpp
#include "improc/threading/thread_pool.hpp"
using namespace improc::threading;

ThreadPool pool(std::thread::hardware_concurrency());

// submit() returns std::future<T>
auto future = pool.submit([]() -> int {
    // ... some work ...
    return 42;
});

int result = future.get();  // blocks until done

// submit_detached() — fire and forget
pool.submit_detached([]() {
    // ... background work, no result needed ...
});

// ThreadPool destructor drains the queue and joins all workers
```

## VideoWriter Standalone

```cpp
#include "improc/io/video_writer.hpp"
using improc::io::VideoWriter;

// Auto-detects codec from extension: .mp4 → mp4v, .avi → MJPG, .mkv → XVID
VideoWriter writer{"output.mp4"};
writer.fps(30);

for (const auto& frame : frames) {
    writer(frame);  // write a frame directly
}

writer.close();  // or let the destructor handle it
```

## Tips

- `CameraCapture` is non-copyable — pass by reference or store in a unique location.
- `FramePipeline` holds references to `CameraCapture` and `ThreadPool`, not ownership. Both must stay alive for the lifetime of the pipeline.
- `tryPop()` never blocks — poll it in your display loop and skip frames if the pool is behind.
- If processing is faster than capture, the pool workers will idle; if slower, the result queue grows. Tune the pool size and pipeline complexity for your target frame rate.

## Next Steps

- [NAMESPACES.md](../../NAMESPACES.md) — complete API reference for every class, method, and error code
- [Building a Pipeline](building-a-pipeline.md) — core ops, format conversions, lazy views, augmentation
