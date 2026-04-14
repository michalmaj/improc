// src/ml/dnn_forward.cpp
#include "improc/ml/dnn_forward.hpp"
#include <algorithm>
#include <filesystem>

namespace improc::ml {

const std::unordered_set<std::string>& DnnForward::valid_extensions() {
    static const std::unordered_set<std::string> exts = {
        ".onnx", ".pb", ".caffemodel", ".weights", ".t7", ".net"
    };
    return exts;
}

DnnForward::DnnForward(std::string model_path) {
    if (!std::filesystem::exists(model_path))
        throw std::runtime_error("DnnForward: model file not found: " + model_path);

    std::string ext = std::filesystem::path(model_path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (!valid_extensions().contains(ext))
        throw std::runtime_error("DnnForward: unsupported model extension '" + ext + "': " + model_path);

    try {
        net_ = cv::dnn::readNet(model_path);
    } catch (const cv::Exception& e) {
        throw std::runtime_error("DnnForward: failed to load model: " + std::string(e.what()));
    }
    if (net_.empty())
        throw std::runtime_error("DnnForward: loaded model is empty: " + model_path);
}

std::vector<float> DnnForward::operator()(const Image<BGR>& img) const {
    cv::Mat blob = cv::dnn::blobFromImage(
        img.mat(), scale_, {input_w_, input_h_}, mean_, swap_rb_);
    net_.setInput(blob);
    cv::Mat output = net_.forward();
    return std::vector<float>(output.begin<float>(), output.end<float>());
}

} // namespace improc::ml
