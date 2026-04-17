// src/ml/dnn_forward.cpp
#include "improc/ml/dnn_forward.hpp"
#include <algorithm>
#include <filesystem>
#include "improc/exceptions.hpp"

using improc::ModelError;

namespace improc::ml {

const std::unordered_set<std::string>& DnnForward::valid_extensions() {
    static const std::unordered_set<std::string> exts = {
        ".onnx", ".pb", ".caffemodel", ".weights", ".t7", ".net"
    };
    return exts;
}

DnnForward::DnnForward(std::string model_path) {
    if (!std::filesystem::exists(model_path))
        throw ModelError{model_path, "file not found"};

    std::string ext = std::filesystem::path(model_path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (!valid_extensions().contains(ext))
        throw ModelError{model_path, "unsupported extension '" + ext + "'"};

    try {
        net_ = cv::dnn::readNet(model_path);
    } catch (const cv::Exception& e) {
        throw ModelError{model_path, "failed to load: " + std::string(e.what())};
    }
    if (net_.empty())
        throw ModelError{model_path, "loaded model is empty"};
}

std::vector<float> DnnForward::operator()(const Image<BGR>& img) const {
    cv::Mat blob = cv::dnn::blobFromImage(
        img.mat(), scale_, {input_w_, input_h_}, mean_, swap_rb_);
    net_.setInput(blob);
    cv::Mat output = net_.forward();
    return std::vector<float>(output.begin<float>(), output.end<float>());
}

} // namespace improc::ml
