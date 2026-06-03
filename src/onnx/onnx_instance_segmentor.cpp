// src/onnx/onnx_instance_segmentor.cpp
#include "improc/onnx/onnx_instance_segmentor.hpp"

#include <cmath>
#include <format>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/exceptions.hpp"

namespace improc::onnx {

// ── Constructor ───────────────────────────────────────────────────────────────

OnnxInstanceSegmentor::OnnxInstanceSegmentor(const std::filesystem::path& path) {
    auto result = session_.load(path);
    if (!result)
        throw improc::ModelError{path, result.error().message};
}

// ── Fluent setters ────────────────────────────────────────────────────────────

OnnxInstanceSegmentor& OnnxInstanceSegmentor::input_size(int w, int h) {
    if (w <= 0 || h <= 0)
        throw improc::ParameterError{"input_size", "dimensions must be positive",
                                     "OnnxInstanceSegmentor"};
    input_w_ = w; input_h_ = h; return *this;
}

OnnxInstanceSegmentor& OnnxInstanceSegmentor::mean(float b, float g, float r) {
    mean_b_ = b; mean_g_ = g; mean_r_ = r; return *this;
}

OnnxInstanceSegmentor& OnnxInstanceSegmentor::scale(float s) {
    if (s <= 0.0f)
        throw improc::ParameterError{"scale", "must be positive", "OnnxInstanceSegmentor"};
    scale_ = s; return *this;
}

OnnxInstanceSegmentor& OnnxInstanceSegmentor::swap_rb(bool v) {
    swap_rb_ = v; return *this;
}

OnnxInstanceSegmentor& OnnxInstanceSegmentor::labels(std::vector<std::string> lbls) {
    labels_ = std::move(lbls); return *this;
}

OnnxInstanceSegmentor& OnnxInstanceSegmentor::confidence_threshold(float t) {
    if (t < 0.0f || t > 1.0f)
        throw improc::ParameterError{"confidence_threshold", "must be in [0, 1]",
                                     "OnnxInstanceSegmentor"};
    confidence_threshold_ = t; return *this;
}

OnnxInstanceSegmentor& OnnxInstanceSegmentor::nms_threshold(float t) {
    if (t < 0.0f || t > 1.0f)
        throw improc::ParameterError{"nms_threshold", "must be in [0, 1]",
                                     "OnnxInstanceSegmentor"};
    nms_threshold_ = t; return *this;
}

OnnxInstanceSegmentor& OnnxInstanceSegmentor::mask_threshold(float t) {
    if (t < 0.0f || t > 1.0f)
        throw improc::ParameterError{"mask_threshold", "must be in [0, 1]",
                                     "OnnxInstanceSegmentor"};
    mask_threshold_ = t; return *this;
}

// ── Inference ─────────────────────────────────────────────────────────────────

std::expected<std::vector<SegmentInstance>, improc::Error>
OnnxInstanceSegmentor::operator()(const Image<BGR>& img) const {
    const int orig_w = img.cols();
    const int orig_h = img.rows();

    // Preprocess (identical to OnnxDetector / OnnxSegmentor)
    cv::Mat resized;
    cv::resize(img.mat(), resized, cv::Size(input_w_, input_h_));

    cv::Mat float_mat;
    resized.convertTo(float_mat, CV_32F, scale_);
    float_mat -= cv::Scalar(mean_b_, mean_g_, mean_r_);

    if (swap_rb_)
        cv::cvtColor(float_mat, float_mat, cv::COLOR_BGR2RGB);

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
    if (outputs.size() < 2)
        return std::unexpected(improc::Error::onnx_inference_failed(
            std::format("YOLO-seg requires 2 output tensors, got {}", outputs.size())));

    const auto& det_out   = outputs[0]; // [1, 4+C+32, N]
    const auto& proto_out = outputs[1]; // [1, 32, Hm, Wm]

    if (det_out.shape.size() < 3 || proto_out.shape.size() < 4)
        return std::unexpected(improc::Error::onnx_inference_failed(
            "unexpected output tensor dimensions"));

    const int64_t features   = det_out.shape[1];
    const int64_t num_boxes  = det_out.shape[2];
    const int     num_masks  = 32;
    const int     num_classes = static_cast<int>(features) - 4 - num_masks;

    if (num_classes <= 0)
        return std::unexpected(improc::Error::onnx_inference_failed(
            std::format("det_out feature count {} too small (need > 36)", features)));

    const int64_t Hm        = proto_out.shape[2];
    const int64_t Wm        = proto_out.shape[3];
    const int     proto_size = static_cast<int>(Hm * Wm);

    const float x_scale = static_cast<float>(orig_w) / static_cast<float>(input_w_);
    const float y_scale = static_cast<float>(orig_h) / static_cast<float>(input_h_);

    // ── Parse detections (YOLOv8 transposed layout: [4+C+32, N]) ────────────
    std::vector<cv::Rect>           boxes;
    std::vector<float>              confs;
    std::vector<int>                class_ids;
    std::vector<std::vector<float>> mask_coeffs;

    for (int64_t i = 0; i < num_boxes; ++i) {
        int   best_class = 0;
        float best_score = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            float score = det_out.data[static_cast<std::size_t>((4 + c) * num_boxes + i)];
            if (score > best_score) { best_score = score; best_class = c; }
        }
        if (best_score < confidence_threshold_) continue;

        const float cx = det_out.data[static_cast<std::size_t>(0 * num_boxes + i)] * x_scale;
        const float cy = det_out.data[static_cast<std::size_t>(1 * num_boxes + i)] * y_scale;
        const float bw = det_out.data[static_cast<std::size_t>(2 * num_boxes + i)] * x_scale;
        const float bh = det_out.data[static_cast<std::size_t>(3 * num_boxes + i)] * y_scale;

        boxes.push_back({static_cast<int>(cx - bw / 2), static_cast<int>(cy - bh / 2),
                         static_cast<int>(bw), static_cast<int>(bh)});
        confs.push_back(best_score);
        class_ids.push_back(best_class);

        std::vector<float> coeff(num_masks);
        for (int m = 0; m < num_masks; ++m)
            coeff[m] = det_out.data[
                static_cast<std::size_t>((4 + num_classes + m) * num_boxes + i)];
        mask_coeffs.push_back(std::move(coeff));
    }

    std::vector<int> nms_idx;
    cv::dnn::NMSBoxes(boxes, confs, confidence_threshold_, nms_threshold_, nms_idx);

    // ── Decode prototype masks for each NMS survivor ─────────────────────────
    std::vector<SegmentInstance> results;
    results.reserve(nms_idx.size());

    for (int idx : nms_idx) {
        // mask_logits[p] = sum_m( coeff[m] * proto[m, p] )
        std::vector<float> mask_logits(static_cast<std::size_t>(proto_size), 0.0f);
        const auto& coeff = mask_coeffs[static_cast<std::size_t>(idx)];
        for (int m = 0; m < num_masks; ++m) {
            const float* proto_row =
                proto_out.data.data() + static_cast<std::size_t>(m * proto_size);
            for (int p = 0; p < proto_size; ++p)
                mask_logits[static_cast<std::size_t>(p)] +=
                    coeff[static_cast<std::size_t>(m)] * proto_row[p];
        }

        // Sigmoid + binarise at mask_threshold_
        cv::Mat mask_small(static_cast<int>(Hm), static_cast<int>(Wm), CV_8U);
        for (int p = 0; p < proto_size; ++p) {
            float sigmoid_val = 1.0f / (1.0f + std::exp(-mask_logits[static_cast<std::size_t>(p)]));
            mask_small.data[p] = (sigmoid_val >= mask_threshold_) ? 255u : 0u;
        }

        // Resize to original image resolution + re-binarise after INTER_LINEAR
        cv::Mat mask_full;
        cv::resize(mask_small, mask_full, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_LINEAR);
        cv::threshold(mask_full, mask_full, 127, 255, cv::THRESH_BINARY);

        int   cid  = class_ids[static_cast<std::size_t>(idx)];
        float conf = confs[static_cast<std::size_t>(idx)];
        std::string lbl = (cid < static_cast<int>(labels_.size()))
                              ? labels_[static_cast<std::size_t>(cid)] : "";

        SegmentInstance inst{
            cv::Rect2f(boxes[static_cast<std::size_t>(idx)]),
            cid,
            conf,
            std::move(lbl),
            improc::core::Image<improc::core::Gray>{mask_full}
        };
        results.push_back(std::move(inst));
    }

    return results;
}

} // namespace improc::onnx
