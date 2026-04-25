// src/onnx/onnx_classifier.cpp
#include "improc/onnx/onnx_classifier.hpp"

#include <algorithm>
#include <opencv2/imgproc.hpp>
#include "improc/exceptions.hpp"

namespace improc::onnx {

// ── Constructor ───────────────────────────────────────────────────────────────

OnnxClassifier::OnnxClassifier(const std::filesystem::path& path) {
    auto result = session_.load(path);
    if (!result)
        throw improc::ModelError{path, result.error().message};
}

// ── Fluent setters ────────────────────────────────────────────────────────────

OnnxClassifier& OnnxClassifier::top_k(int k) {
    if (k <= 0) throw improc::ParameterError{"top_k", "must be positive", "OnnxClassifier"};
    top_k_ = k;
    return *this;
}

OnnxClassifier& OnnxClassifier::input_size(int w, int h) {
    if (w <= 0 || h <= 0)
        throw improc::ParameterError{"input_size", "dimensions must be positive", "OnnxClassifier"};
    input_w_ = w;
    input_h_ = h;
    return *this;
}

OnnxClassifier& OnnxClassifier::mean(float b, float g, float r) {
    mean_b_ = b; mean_g_ = g; mean_r_ = r;
    return *this;
}

OnnxClassifier& OnnxClassifier::scale(float s) {
    if (s <= 0.0f) throw improc::ParameterError{"scale", "must be positive", "OnnxClassifier"};
    scale_ = s;
    return *this;
}

OnnxClassifier& OnnxClassifier::swap_rb(bool v) {
    swap_rb_ = v;
    return *this;
}

OnnxClassifier& OnnxClassifier::labels(std::vector<std::string> lbls) {
    labels_ = std::move(lbls);
    return *this;
}

// ── Inference ─────────────────────────────────────────────────────────────────

std::expected<std::vector<ClassResult>, improc::Error>
OnnxClassifier::operator()(const Image<BGR>& img) const {
    // Resize
    cv::Mat resized;
    cv::resize(img.mat(), resized, cv::Size(input_w_, input_h_));

    // Float conversion + scale
    cv::Mat float_mat;
    resized.convertTo(float_mat, CV_32F, scale_);

    // Mean subtraction (BGR order)
    float_mat -= cv::Scalar(mean_b_, mean_g_, mean_r_);

    // Optional BGR→RGB swap
    if (swap_rb_)
        cv::cvtColor(float_mat, float_mat, cv::COLOR_BGR2RGB);

    // HWC → CHW: split channels, copy into flat tensor
    std::vector<cv::Mat> channels;
    cv::split(float_mat, channels);

    std::vector<float> data;
    data.reserve(static_cast<size_t>(3 * input_h_ * input_w_));
    for (auto& ch : channels) {
        const float* ptr = ch.ptr<float>(0);
        data.insert(data.end(), ptr, ptr + input_h_ * input_w_);
    }

    // Run session
    auto input_names = session_.input_names();
    if (input_names.empty())
        return std::unexpected(improc::Error::onnx_session_not_loaded());

    TensorInfo input{
        input_names[0],
        {1, 3, static_cast<int64_t>(input_h_), static_cast<int64_t>(input_w_)},
        std::move(data)
    };

    auto run_result = session_.run({input});
    if (!run_result) return std::unexpected(run_result.error());

    const auto& output_data = run_result.value()[0].data;

    // Collect (score, index) pairs and partial-sort for top-k
    std::vector<std::pair<float, int>> scores;
    scores.reserve(output_data.size());
    for (int i = 0; i < static_cast<int>(output_data.size()); ++i)
        scores.emplace_back(output_data[i], i);

    int k = std::min(top_k_, static_cast<int>(scores.size()));
    std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<ClassResult> results;
    results.reserve(static_cast<size_t>(k));
    for (int i = 0; i < k; ++i) {
        auto [score, idx] = scores[i];
        std::string lbl = (idx < static_cast<int>(labels_.size())) ? labels_[idx] : "";
        results.push_back({idx, score, std::move(lbl)});
    }

    return results;
}

} // namespace improc::onnx
