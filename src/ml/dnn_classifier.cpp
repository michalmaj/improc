// src/ml/dnn_classifier.cpp
#include "improc/ml/dnn_classifier.hpp"

namespace improc::ml {

const std::unordered_set<std::string>& DnnClassifier::valid_extensions() {
    static const std::unordered_set<std::string> exts = {
        ".onnx", ".pb", ".caffemodel", ".weights", ".t7", ".net"
    };
    return exts;
}

DnnClassifier::DnnClassifier(std::string model_path) {
    std::filesystem::path p{model_path};
    if (!std::filesystem::exists(p))
        throw ModelError{p, "file not found"};

    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (!valid_extensions().contains(ext))
        throw ModelError{p, "unsupported extension '" + ext + "'"};

    try {
        net_ = cv::dnn::readNet(model_path);
    } catch (const cv::Exception& e) {
        throw ModelError{p, "failed to parse model: " + std::string(e.what())};
    }
    if (net_.empty())
        throw ModelError{p, "loaded model is empty"};
}

std::vector<ClassResult> DnnClassifier::operator()(const Image<BGR>& img) const {
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
