// examples/ml/demo_onnx_inference.cpp
//
// Demonstrates improc::onnx: OnnxSession, OnnxClassifier, OnnxDetector.
//
// Part 1 — always works (no external files required):
//   Uses the tiny synthetic models from tests/onnx/testdata/ to show
//   each layer of the API: raw OnnxSession, OnnxClassifier, OnnxDetector.
//   Run from the project root so relative paths resolve correctly.
//
// Part 2 — real-model pattern (optional, requires external .onnx files):
//   Shows the configuration needed for MobileNetV2 classification and
//   YOLOv8n detection. Activate by passing paths as command-line arguments.
//
// Build:  cmake --build build --target demo_onnx_inference
// Run:    ./build/demo_onnx_inference
//   or:   ./build/demo_onnx_inference /path/to/mobilenet.onnx /path/to/yolov8n.onnx

#include <iostream>
#include <format>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include "improc/onnx/onnx.hpp"
#include "improc/core/image.hpp"

using improc::core::Image;
using improc::core::BGR;
using improc::onnx::OnnxSession;
using improc::onnx::OnnxClassifier;
using improc::onnx::OnnxDetector;
using improc::onnx::TensorInfo;

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

static Image<BGR> make_solid_image(int w, int h, cv::Scalar color) {
    cv::Mat mat(h, w, CV_8UC3, color);
    return Image<BGR>(mat);
}

static void print_separator(const std::string& title) {
    std::cout << "\n--- " << title << " ---\n";
}

// --------------------------------------------------------------------------
// Part 1a: Raw OnnxSession
// --------------------------------------------------------------------------

static void demo_raw_session(const std::string& model_path) {
    print_separator("OnnxSession (raw tensor API)");

    OnnxSession session;
    auto load_result = session.load(model_path);
    if (!load_result) {
        std::cerr << "  Load failed: " << load_result.error().message << "\n";
        return;
    }
    std::cout << std::format("  Model:   {}\n", model_path);
    std::cout << std::format("  Inputs:  {}\n", session.input_names()[0]);
    std::cout << std::format("  Outputs: {}\n", session.output_names()[0]);

    // Build a flat [1, 3, 8, 8] tensor filled with 0.5
    constexpr int N = 1 * 3 * 8 * 8;
    TensorInfo input{session.input_names()[0], {1, 3, 8, 8}, std::vector<float>(N, 0.5f)};

    auto result = session.run({input});
    if (!result) {
        std::cerr << "  Inference failed: " << result.error().message << "\n";
        return;
    }
    const auto& out = result->front();
    std::cout << std::format("  Output shape: [{}]\n",
        [&]{ std::string s;
             for (auto d : out.shape) s += std::to_string(d) + ", ";
             if (!s.empty()) s.resize(s.size() - 2);
             return s; }());
    std::cout << std::format("  Output[0..2]: {:.4f}  {:.4f}  {:.4f}\n",
                             out.data[0], out.data[1], out.data[2]);
}

// --------------------------------------------------------------------------
// Part 1b: OnnxClassifier with tiny model
// --------------------------------------------------------------------------

static void demo_classifier_tiny(const std::string& model_path) {
    print_separator("OnnxClassifier (tiny synthetic model, 3 classes)");

    OnnxClassifier cls{model_path};
    cls.input_size(8, 8)   // tiny model expects 8×8 input
       .top_k(3)
       .scale(1.0f / 255.0f)
       .swap_rb(true)
       .labels({"class_A", "class_B", "class_C"});

    Image<BGR> img = make_solid_image(64, 64, {128, 64, 200});
    auto result = cls(img);
    if (!result) {
        std::cerr << "  Inference failed: " << result.error().message << "\n";
        return;
    }
    std::cout << "  Top results:\n";
    for (const auto& r : *result)
        std::cout << std::format("    [{:7.4f}]  {}\n", r.score, r.label);
}

// --------------------------------------------------------------------------
// Part 1c: OnnxDetector with tiny YOLOv8 model
// --------------------------------------------------------------------------

static void demo_detector_tiny(const std::string& model_path) {
    print_separator("OnnxDetector (tiny synthetic YOLOv8 model, 2 classes)");

    OnnxDetector det{model_path};
    det.input_size(8, 8)
       .confidence_threshold(0.01f)   // very low — synthetic weights produce near-zero scores
       .nms_threshold(0.4f)
       .labels({"cat", "dog"});

    Image<BGR> img = make_solid_image(320, 240, {50, 180, 80});
    auto result = det(img);
    if (!result) {
        std::cerr << "  Inference failed: " << result.error().message << "\n";
        return;
    }
    if (result->empty())
        std::cout << "  No detections above threshold (expected with random weights).\n";
    else
        for (const auto& d : *result)
            std::cout << std::format("  [{:.3f}] {} @ ({},{}) {}x{}\n",
                d.confidence, d.label, d.box.x, d.box.y, d.box.width, d.box.height);
}

// --------------------------------------------------------------------------
// Part 2: Real-model patterns (skipped unless paths supplied)
// --------------------------------------------------------------------------

static void demo_real_classifier(const std::string& path) {
    print_separator("OnnxClassifier (real model — MobileNetV2 pattern)");

    // MobileNetV2 from ONNX Model Zoo:
    //   python -c "import urllib.request; urllib.request.urlretrieve(
    //     'https://github.com/onnx/models/raw/main/validated/vision/classification/'
    //     'mobilenet/model/mobilenetv2-12.onnx', 'mobilenetv2-12.onnx')"
    //
    // ImageNet normalization: mean=[0.485,0.456,0.406] × 255, scale=1/255, RGB input.
    // Note: mean is in R,G,B order here — pass as B,G,R to match improc convention.

    OnnxClassifier cls{path};
    cls.input_size(224, 224)
       .scale(1.0f / 255.0f)
       .mean(0.406f * 255.0f, 0.456f * 255.0f, 0.485f * 255.0f)  // B, G, R
       .swap_rb(true)
       .top_k(5);

    Image<BGR> img = make_solid_image(640, 480, {100, 150, 200});
    auto result = cls(img);
    if (!result) { std::cerr << "  " << result.error().message << "\n"; return; }

    std::cout << "  Top-5 (no labels loaded — indices shown):\n";
    for (const auto& r : *result)
        std::cout << std::format("    [{:.4f}]  {}\n", r.score, r.label);
}

static void demo_real_detector(const std::string& path) {
    print_separator("OnnxDetector (real model — YOLOv8n pattern)");

    // Export with: model = YOLO('yolov8n.pt'); model.export(format='onnx', opset=12)
    // For ORT 1.20.1 compatibility set IR version: import onnx; m=onnx.load(...);
    //   m.ir_version=7; onnx.save(m, 'yolov8n_ir7.onnx')

    OnnxDetector det{path};
    det.input_size(640, 640)
       .confidence_threshold(0.5f)
       .nms_threshold(0.4f)
       .swap_rb(true);

    Image<BGR> img = make_solid_image(1280, 720, {60, 120, 180});
    auto result = det(img);
    if (!result) { std::cerr << "  " << result.error().message << "\n"; return; }

    std::cout << std::format("  Detections: {}\n", result->size());
    for (const auto& d : *result)
        std::cout << std::format("  [{:.3f}] {} @ ({},{}) {}x{}\n",
            d.confidence, d.label, d.box.x, d.box.y, d.box.width, d.box.height);
}

// --------------------------------------------------------------------------
// main
// --------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    std::cout << "=== improc::onnx inference demo ===\n";

    const std::string cls_tiny  = "tests/onnx/testdata/tiny_classifier.onnx";
    const std::string det_tiny  = "tests/onnx/testdata/tiny_detector_yolov8.onnx";

    // --- Part 1: always-on, tiny synthetic models ---
    std::cout << "\n[Part 1] Tiny synthetic models (no external files required)\n";
    std::cout << "  Run from the project root so relative paths resolve.\n";

    demo_raw_session(cls_tiny);
    demo_classifier_tiny(cls_tiny);
    demo_detector_tiny(det_tiny);

    // --- Part 2: real models (optional, path supplied as argv) ---
    if (argc >= 2) {
        std::cout << "\n[Part 2] Real classifier model: " << argv[1] << "\n";
        demo_real_classifier(argv[1]);
    }
    if (argc >= 3) {
        std::cout << "\n[Part 2] Real detector model: " << argv[2] << "\n";
        demo_real_detector(argv[2]);
    }
    if (argc < 2) {
        std::cout << "\n[Part 2] Skipped — pass .onnx paths to activate:\n";
        std::cout << "  ./build/demo_onnx_inference mobilenet.onnx yolov8n.onnx\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
