// src/ml/dnn_segmentor.cpp
#include "improc/ml/dnn_segmentor.hpp"
#include <format>
#include <opencv2/imgproc.hpp>

using improc::ModelError;

namespace improc::ml {

DnnSegmentor::DnnSegmentor(const std::filesystem::path& model,
                           const std::filesystem::path& config)
{
    if (!std::filesystem::exists(model))
        throw ModelError{model.string(), "file not found"};

    try {
        net_ = cv::dnn::readNet(model.string(),
                                config.empty() ? "" : config.string());
    } catch (const cv::Exception& e) {
        throw ModelError{model.string(), "cv::dnn failed to load: " + std::string(e.what())};
    }
    if (net_.empty())
        throw ModelError{model.string(), "loaded model is empty"};
}

std::expected<SegmentationMask, improc::Error>
DnnSegmentor::operator()(const Image<BGR>& img) const
{
    const int orig_w = img.cols();
    const int orig_h = img.rows();

    cv::Mat blob = cv::dnn::blobFromImage(
        img.mat(), scale_, {input_w_, input_h_},
        {mean_b_, mean_g_, mean_r_}, swap_rb_);
    net_.setInput(blob);

    cv::Mat output;
    try {
        output = output_layer_.empty()
            ? net_.forward()
            : net_.forward(output_layer_);
    } catch (const cv::Exception& e) {
        return std::unexpected(improc::Error{
            improc::Error::Code::OnnxInferenceFailed,
            std::format("DnnSegmentor: forward pass failed: {}", e.what())});
    }

    if (output.dims != 4) {
        return std::unexpected(improc::Error{
            improc::Error::Code::OnnxInferenceFailed,
            std::format("DnnSegmentor: expected 4-D output [1,C,H,W], got {} dims",
                        output.dims)});
    }

    const int C    = output.size[1];
    const int H    = output.size[2];
    const int W    = output.size[3];
    const auto* data = reinterpret_cast<const float*>(output.data);

    cv::Mat mask(H, W, CV_8UC1);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int   best_c   = 0;
            float best_val = data[y * W + x];          // c=0
            for (int c = 1; c < C; ++c) {
                float val = data[c * H * W + y * W + x];
                if (val > best_val) { best_val = val; best_c = c; }
            }
            mask.at<uchar>(y, x) = static_cast<uchar>(best_c);
        }
    }

    cv::Mat mask_resized;
    cv::resize(mask, mask_resized, {orig_w, orig_h}, 0, 0, cv::INTER_NEAREST);
    return SegmentationMask{Image<Gray>{mask_resized}, labels_};
}

} // namespace improc::ml
