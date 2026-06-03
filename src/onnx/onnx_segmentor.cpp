// src/onnx/onnx_segmentor.cpp
#include "improc/onnx/onnx_segmentor.hpp"

#include <format>
#include <opencv2/imgproc.hpp>
#include "improc/exceptions.hpp"

namespace improc::onnx {

// ── Constructor ───────────────────────────────────────────────────────────────

OnnxSegmentor::OnnxSegmentor(const std::filesystem::path& path) {
    auto result = session_.load(path);
    if (!result)
        throw improc::ModelError{path, result.error().message};
}

// ── Fluent setters ────────────────────────────────────────────────────────────

OnnxSegmentor& OnnxSegmentor::input_size(int w, int h) {
    if (w <= 0 || h <= 0)
        throw improc::ParameterError{"input_size", "dimensions must be positive", "OnnxSegmentor"};
    input_w_ = w; input_h_ = h; return *this;
}

OnnxSegmentor& OnnxSegmentor::mean(float b, float g, float r) {
    mean_b_ = b; mean_g_ = g; mean_r_ = r; return *this;
}

OnnxSegmentor& OnnxSegmentor::scale(float s) {
    if (s <= 0.0f)
        throw improc::ParameterError{"scale", "must be positive", "OnnxSegmentor"};
    scale_ = s; return *this;
}

OnnxSegmentor& OnnxSegmentor::swap_rb(bool v) { swap_rb_ = v; return *this; }

OnnxSegmentor& OnnxSegmentor::labels(std::vector<std::string> lbls) {
    labels_ = std::move(lbls); return *this;
}

// ── Inference ─────────────────────────────────────────────────────────────────

std::expected<SegmentationMask, improc::Error>
OnnxSegmentor::operator()(const Image<BGR>& img) const {
    const int orig_w = img.cols();
    const int orig_h = img.rows();

    // Preprocess: resize → float → scale → mean subtract → optional BGR→RGB
    cv::Mat resized;
    cv::resize(img.mat(), resized, cv::Size(input_w_, input_h_));

    cv::Mat float_mat;
    resized.convertTo(float_mat, CV_32F, scale_);
    float_mat -= cv::Scalar(mean_b_, mean_g_, mean_r_);

    if (swap_rb_)
        cv::cvtColor(float_mat, float_mat, cv::COLOR_BGR2RGB);

    // HWC → CHW: split channels then lay them contiguously
    std::vector<cv::Mat> channels;
    cv::split(float_mat, channels);

    std::vector<float> data;
    data.reserve(static_cast<std::size_t>(3 * input_h_ * input_w_));
    for (auto& ch : channels) {
        const float* ptr = ch.ptr<float>(0);
        data.insert(data.end(), ptr, ptr + input_h_ * input_w_);
    }

    auto input_names = session_.input_names();
    if (input_names.empty())
        return std::unexpected(improc::Error::onnx_session_not_loaded());

    TensorInfo input_tensor{
        input_names[0],
        {1, 3, static_cast<int64_t>(input_h_), static_cast<int64_t>(input_w_)},
        std::move(data)
    };

    auto run_result = session_.run({input_tensor});
    if (!run_result) return std::unexpected(run_result.error());

    const auto& outputs = run_result.value();
    if (outputs.empty())
        return std::unexpected(improc::Error::onnx_inference_failed(
            "model produced no output tensors"));

    // Expect [1, C, H, W]
    const auto& out = outputs[0];
    if (out.shape.size() < 4)
        return std::unexpected(improc::Error::onnx_inference_failed(
            std::format("expected 4-D output [1,C,H,W], got {} dims", out.shape.size())));

    const int C  = static_cast<int>(out.shape[1]);
    const int OH = static_cast<int>(out.shape[2]);
    const int OW = static_cast<int>(out.shape[3]);

    // Argmax over C at each spatial position → CV_8U class-index mask
    cv::Mat mask(OH, OW, CV_8U);
    for (int y = 0; y < OH; ++y) {
        for (int x = 0; x < OW; ++x) {
            int   best_c   = 0;
            float best_val = out.data[static_cast<std::size_t>(0 * OH * OW + y * OW + x)];
            for (int c = 1; c < C; ++c) {
                float val = out.data[static_cast<std::size_t>(c * OH * OW + y * OW + x)];
                if (val > best_val) { best_val = val; best_c = c; }
            }
            mask.at<uint8_t>(y, x) = static_cast<uint8_t>(best_c);
        }
    }

    // Resize back to original resolution; INTER_NEAREST preserves integer class IDs
    cv::Mat mask_resized;
    cv::resize(mask, mask_resized, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_NEAREST);

    return SegmentationMask{improc::core::Image<improc::core::Gray>{mask_resized}, labels_};
}

} // namespace improc::onnx
