// src/ml/dnn_classifier.cpp
#include "improc/ml/dnn_classifier.hpp"
#include <algorithm>
#include <fstream>
#include <opencv2/imgproc.hpp>

namespace improc::ml {

const std::vector<std::string>& DnnClassifier::valid_extensions() {
    static const std::vector<std::string> exts = {
        ".onnx", ".pb", ".caffemodel", ".weights", ".t7", ".net"
    };
    return exts;
}

DnnClassifier::DnnClassifier(std::string model_path) {
    if (!std::filesystem::exists(model_path))
        throw std::runtime_error("DnnClassifier: model file not found: " + model_path);

    std::string ext = std::filesystem::path(model_path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    const auto& valid = valid_extensions();
    if (std::find(valid.begin(), valid.end(), ext) == valid.end())
        throw std::runtime_error("DnnClassifier: unsupported model extension '" + ext + "': " + model_path);

    try {
        net_ = cv::dnn::readNet(model_path);
    } catch (const cv::Exception& e) {
        throw std::runtime_error("DnnClassifier: failed to load model: " + std::string(e.what()));
    }
    if (net_.empty())
        throw std::runtime_error("DnnClassifier: loaded model is empty: " + model_path);
}

std::vector<ClassResult> DnnClassifier::operator()(Image<BGR> img) const {
    cv::Mat blob = cv::dnn::blobFromImage(
        img.mat(), scale_, {input_w_, input_h_}, mean_, swap_rb_);
    net_.setInput(blob);
    cv::Mat output = net_.forward();

    // Flatten to [1, num_classes]
    cv::Mat scores = output.reshape(1, 1);

    cv::Mat sorted_idx;
    cv::sortIdx(scores, sorted_idx, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);

    std::vector<ClassResult> results;
    int k = std::min(top_k_, scores.cols);
    results.reserve(k);
    for (int i = 0; i < k; ++i) {
        int idx = sorted_idx.at<int>(0, i);
        ClassResult r;
        r.class_id = idx;
        r.score    = scores.at<float>(0, idx);
        r.label    = (idx < static_cast<int>(labels_.size())) ? labels_[idx] : "";
        results.push_back(std::move(r));
    }
    return results;
}

} // namespace improc::ml
