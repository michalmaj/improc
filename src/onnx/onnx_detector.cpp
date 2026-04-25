// src/onnx/onnx_detector.cpp
#include "improc/onnx/onnx_detector.hpp"

#include <algorithm>
#include <format>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/exceptions.hpp"

namespace improc::onnx {

// ── Constructor ───────────────────────────────────────────────────────────────

OnnxDetector::OnnxDetector(const std::filesystem::path& path) {
    auto result = session_.load(path);
    if (!result)
        throw improc::ModelError{path, result.error().message};
}

// ── Fluent setters ────────────────────────────────────────────────────────────

OnnxDetector& OnnxDetector::style(Style s) {
    style_ = s; return *this;
}

OnnxDetector& OnnxDetector::confidence_threshold(float t) {
    if (t < 0.0f || t > 1.0f)
        throw improc::ParameterError{"confidence_threshold", "must be in [0, 1]", "OnnxDetector"};
    confidence_threshold_ = t; return *this;
}

OnnxDetector& OnnxDetector::nms_threshold(float t) {
    if (t < 0.0f || t > 1.0f)
        throw improc::ParameterError{"nms_threshold", "must be in [0, 1]", "OnnxDetector"};
    nms_threshold_ = t; return *this;
}

OnnxDetector& OnnxDetector::input_size(int w, int h) {
    if (w <= 0 || h <= 0)
        throw improc::ParameterError{"input_size", "dimensions must be positive", "OnnxDetector"};
    input_w_ = w; input_h_ = h; return *this;
}

OnnxDetector& OnnxDetector::mean(float b, float g, float r) {
    mean_b_ = b; mean_g_ = g; mean_r_ = r; return *this;
}

OnnxDetector& OnnxDetector::scale(float s) {
    if (s <= 0.0f)
        throw improc::ParameterError{"scale", "must be positive", "OnnxDetector"};
    scale_ = s; return *this;
}

OnnxDetector& OnnxDetector::swap_rb(bool v) {
    swap_rb_ = v; return *this;
}

OnnxDetector& OnnxDetector::labels(std::vector<std::string> lbls) {
    labels_ = std::move(lbls); return *this;
}

// ── Inference ─────────────────────────────────────────────────────────────────

std::expected<std::vector<Detection>, improc::Error>
OnnxDetector::operator()(const Image<BGR>& img) const {
    const int orig_w = img.cols();
    const int orig_h = img.rows();

    // Preprocess
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

    const auto& outputs = run_result.value();
    if (outputs.empty())
        return std::unexpected(improc::Error::onnx_inference_failed("model produced no output tensors"));

    if (style_ == Style::YOLO)
        return parse_yolo(outputs[0], orig_w, orig_h);

    if (outputs.size() < 2)
        return std::unexpected(improc::Error::onnx_inference_failed(
            "SSD style requires 2 output tensors, got " + std::to_string(outputs.size())));

    return parse_ssd(outputs[0], outputs[1], orig_w, orig_h);
}

// ── YOLO post-processing ──────────────────────────────────────────────────────

std::vector<Detection>
OnnxDetector::parse_yolo(const TensorInfo& output, int orig_w, int orig_h) const {
    if (output.shape.size() < 3)
        throw improc::Exception{std::format(
            "OnnxDetector: YOLO output must be 3-D, got {} dims", output.shape.size())};

    const int64_t dim1 = output.shape[1];
    const int64_t dim2 = output.shape[2];

    // YOLOv8 exports [1, 4+C, N]; YOLOv5 exports [1, N, 5+C].
    // Heuristic: if dim1 < dim2 the features axis is the smaller one → transposed.
    const bool transposed   = (dim1 < dim2);
    const int64_t num_boxes = transposed ? dim2 : dim1;
    const int64_t features  = transposed ? dim1 : dim2;

    const float x_scale = static_cast<float>(orig_w) / input_w_;
    const float y_scale = static_cast<float>(orig_h) / input_h_;

    std::vector<cv::Rect> boxes;
    std::vector<float>    confs;
    std::vector<int>      class_ids;

    if (transposed) {
        // YOLOv8 layout: [4+C, N] — no objectness score; class scores directly.
        const int num_classes = static_cast<int>(features) - 4;
        if (num_classes <= 0)
            throw improc::Exception{std::format(
                "OnnxDetector: YOLO (transposed) output has unexpected feature count: {}", features)};

        for (int64_t i = 0; i < num_boxes; ++i) {
            int   best_class = 0;
            float best_score = 0.0f;
            for (int c = 0; c < num_classes; ++c) {
                float score = output.data[static_cast<size_t>((4 + c) * num_boxes + i)];
                if (score > best_score) { best_score = score; best_class = c; }
            }
            if (best_score < confidence_threshold_) continue;

            const float cx = output.data[static_cast<size_t>(0 * num_boxes + i)] * x_scale;
            const float cy = output.data[static_cast<size_t>(1 * num_boxes + i)] * y_scale;
            const float w  = output.data[static_cast<size_t>(2 * num_boxes + i)] * x_scale;
            const float h  = output.data[static_cast<size_t>(3 * num_boxes + i)] * y_scale;

            boxes.push_back({static_cast<int>(cx - w / 2), static_cast<int>(cy - h / 2),
                             static_cast<int>(w),           static_cast<int>(h)});
            confs.push_back(best_score);
            class_ids.push_back(best_class);
        }
    } else {
        // YOLOv5 layout: [N, 5+C] — col 4 is objectness score.
        const int num_classes = static_cast<int>(features) - 5;
        if (num_classes <= 0)
            throw improc::Exception{std::format(
                "OnnxDetector: YOLO output has unexpected feature count: {}", features)};

        for (int64_t i = 0; i < num_boxes; ++i) {
            const float* d        = output.data.data() + i * features;
            const float  obj_conf = d[4];
            if (obj_conf < confidence_threshold_) continue;

            int   best_class = 0;
            float best_score = 0.0f;
            for (int c = 0; c < num_classes; ++c) {
                float s = obj_conf * d[5 + c];
                if (s > best_score) { best_score = s; best_class = c; }
            }
            if (best_score < confidence_threshold_) continue;

            const float cx = d[0] * x_scale;
            const float cy = d[1] * y_scale;
            const float w  = d[2] * x_scale;
            const float h  = d[3] * y_scale;

            boxes.push_back({static_cast<int>(cx - w / 2), static_cast<int>(cy - h / 2),
                             static_cast<int>(w),           static_cast<int>(h)});
            confs.push_back(best_score);
            class_ids.push_back(best_class);
        }
    }

    std::vector<int> nms_idx;
    cv::dnn::NMSBoxes(boxes, confs, confidence_threshold_, nms_threshold_, nms_idx);

    std::vector<Detection> results;
    results.reserve(nms_idx.size());
    for (int idx : nms_idx) {
        Detection det;
        det.box        = cv::Rect2f(boxes[idx]);
        det.class_id   = class_ids[idx];
        det.confidence = confs[idx];
        det.label = (det.class_id < static_cast<int>(labels_.size()))
                        ? labels_[det.class_id] : "";
        results.push_back(std::move(det));
    }
    return results;
}

// ── SSD post-processing ───────────────────────────────────────────────────────

std::vector<Detection>
OnnxDetector::parse_ssd(const TensorInfo& boxes_t, const TensorInfo& scores_t,
                         int orig_w, int orig_h) const {
    // boxes_t:  [1, N, 4] — [y1, x1, y2, x2] normalised
    // scores_t: [1, N, C] — per-class scores; index 0 = background
    if (boxes_t.shape.size() < 3 || scores_t.shape.size() < 3)
        throw improc::Exception{"OnnxDetector: SSD output tensors must be 3-D"};

    const int64_t num_dets   = boxes_t.shape[1];
    const int64_t num_classes = scores_t.shape[2];

    if (scores_t.shape[1] != num_dets)
        throw improc::Exception{"OnnxDetector: SSD boxes/scores first dimension mismatch"};

    std::vector<cv::Rect> boxes;
    std::vector<float>    confs;
    std::vector<int>      class_ids;

    for (int64_t i = 0; i < num_dets; ++i) {
        if (num_classes <= 1) continue;

        const float* s = scores_t.data.data() + i * num_classes;
        int   best_class = 1;
        float best_score = s[1];
        for (int64_t c = 2; c < num_classes; ++c) {
            if (s[c] > best_score) { best_score = s[c]; best_class = static_cast<int>(c); }
        }
        if (best_score < confidence_threshold_) continue;

        const float* b = boxes_t.data.data() + i * 4;
        const float y1 = b[0] * orig_h;
        const float x1 = b[1] * orig_w;
        const float y2 = b[2] * orig_h;
        const float x2 = b[3] * orig_w;

        boxes.push_back({static_cast<int>(x1), static_cast<int>(y1),
                         static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)});
        confs.push_back(best_score);
        class_ids.push_back(best_class - 1);  // subtract background offset
    }

    std::vector<int> nms_idx;
    cv::dnn::NMSBoxes(boxes, confs, confidence_threshold_, nms_threshold_, nms_idx);

    std::vector<Detection> results;
    results.reserve(nms_idx.size());
    for (int idx : nms_idx) {
        Detection det;
        det.box        = cv::Rect2f(boxes[idx]);
        det.class_id   = class_ids[idx];
        det.confidence = confs[idx];
        det.label = (det.class_id < static_cast<int>(labels_.size()))
                        ? labels_[det.class_id] : "";
        results.push_back(std::move(det));
    }
    return results;
}

} // namespace improc::onnx
