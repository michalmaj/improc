//
// Created by Michał Maj on 14/04/2026.
//
// Demo: DnnClassifier, DnnDetector, DnnForward
//
// To run inference, place model files in examples/ml/models/:
//   - classifier.onnx  (e.g. ResNet-50 from ONNX Model Zoo)
//   - detector.onnx    (e.g. YOLOv8n from ultralytics)
//   - encoder.onnx     (any encoder/feature extractor)
//
// Download ResNet-50: https://github.com/onnx/models/tree/main/validated/vision/classification/resnet
// Export YOLOv8n:    yolo export model=yolov8n.pt format=onnx  (requires ultralytics)

#include "improc/ml/dnn_classifier.hpp"
#include "improc/ml/dnn_detector.hpp"
#include "improc/ml/dnn_forward.hpp"
#include "improc/core/pipeline.hpp"
#include <filesystem>
#include <iostream>

using namespace improc::core;
using namespace improc::ml;
namespace fs = std::filesystem;

int main() {
    const fs::path models_dir = fs::path(__FILE__).parent_path() / "models";

    // --- 1. Classification ---
    const auto clf_path = models_dir / "classifier.onnx";
    if (fs::exists(clf_path)) {
        std::cout << "--- DnnClassifier ---\n";
        cv::Mat mat(224, 224, CV_8UC3, cv::Scalar(100, 150, 200));
        Image<BGR> img(mat);

        auto results = img | DnnClassifier{clf_path.string()}.top_k(3);
        for (const auto& r : results)
            std::cout << "  class=" << r.class_id
                      << " score=" << r.score
                      << " label=" << r.label << "\n";
    } else {
        std::cout << "Skipping classifier demo (place classifier.onnx in examples/ml/models/)\n";
    }

    // --- 2. Detection ---
    const auto det_path = models_dir / "detector.onnx";
    if (fs::exists(det_path)) {
        std::cout << "--- DnnDetector (YOLO) ---\n";
        cv::Mat mat(640, 640, CV_8UC3, cv::Scalar(80, 80, 80));
        Image<BGR> img(mat);

        auto detections = DnnDetector{det_path.string()}
                              .confidence_threshold(0.5f)
                              .nms_threshold(0.4f)(img);
        std::cout << "  Detections: " << detections.size() << "\n";
        for (const auto& d : detections)
            std::cout << "  class=" << d.class_id
                      << " conf=" << d.confidence
                      << " box=[" << d.box.x << "," << d.box.y
                      << "," << d.box.width << "," << d.box.height << "]\n";
    } else {
        std::cout << "Skipping detector demo (place detector.onnx in examples/ml/models/)\n";
    }

    // --- 3. Raw forward pass ---
    const auto enc_path = models_dir / "encoder.onnx";
    if (fs::exists(enc_path)) {
        std::cout << "--- DnnForward ---\n";
        cv::Mat mat(224, 224, CV_8UC3, cv::Scalar(50, 100, 150));
        Image<BGR> img(mat);

        auto blob = DnnForward{enc_path.string()}(img);
        std::cout << "  Output tensor size: " << blob.size() << "\n";
        if (!blob.empty())
            std::cout << "  First value: " << blob[0] << "\n";
    } else {
        std::cout << "Skipping forward demo (place encoder.onnx in examples/ml/models/)\n";
    }

    std::cout << "Done.\n";
    return 0;
}
