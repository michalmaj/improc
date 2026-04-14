// src/ml/dnn_detector.cpp
#include "improc/ml/dnn_detector.hpp"
#include <algorithm>
#include <filesystem>

namespace improc::ml {

const std::unordered_set<std::string>& DnnDetector::valid_extensions() {
    static const std::unordered_set<std::string> exts = {
        ".onnx", ".pb", ".caffemodel", ".weights", ".t7", ".net"
    };
    return exts;
}

DnnDetector::DnnDetector(std::string model_path) {
    if (!std::filesystem::exists(model_path))
        throw std::runtime_error("DnnDetector: model file not found: " + model_path);

    std::string ext = std::filesystem::path(model_path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (!valid_extensions().contains(ext))
        throw std::runtime_error("DnnDetector: unsupported model extension '" + ext + "': " + model_path);

    try {
        net_ = cv::dnn::readNet(model_path);
    } catch (const cv::Exception& e) {
        throw std::runtime_error("DnnDetector: failed to load model: " + std::string(e.what()));
    }
    if (net_.empty())
        throw std::runtime_error("DnnDetector: loaded model is empty: " + model_path);
}

std::vector<Detection> DnnDetector::operator()(const Image<BGR>& img) const {
    const int orig_w = img.cols();
    const int orig_h = img.rows();

    cv::Mat blob = cv::dnn::blobFromImage(
        img.mat(), scale_, {input_w_, input_h_}, mean_, swap_rb_);
    net_.setInput(blob);

    if (style_ == Style::YOLO) {
        std::vector<cv::Mat> outs;
        if (output_layer_.empty())
            net_.forward(outs, net_.getUnconnectedOutLayersNames());
        else
            outs = {net_.forward(output_layer_)};
        cv::Mat out = outs[0];
        return parse_yolo(out, orig_w, orig_h);
    } else {
        cv::Mat boxes  = net_.forward(boxes_layer_);
        cv::Mat scores = net_.forward(scores_layer_);
        return parse_ssd(boxes, scores, orig_w, orig_h);
    }
}

std::vector<Detection> DnnDetector::parse_yolo(cv::Mat& output,
                                                int orig_w, int orig_h) const {
    // Reshape to [N, 5+num_classes]
    cv::Mat det_mat;
    if (output.dims == 3)
        det_mat = output.reshape(1, output.size[1]);
    else
        det_mat = output;

    const float x_scale = static_cast<float>(orig_w) / input_w_;
    const float y_scale = static_cast<float>(orig_h) / input_h_;

    std::vector<cv::Rect> boxes;
    std::vector<float>    confs;
    std::vector<int>      class_ids;

    for (int i = 0; i < det_mat.rows; ++i) {
        const float* d = det_mat.ptr<float>(i);
        const float obj_conf = d[4];
        if (obj_conf < confidence_threshold_) continue;

        const int num_classes = det_mat.cols - 5;
        cv::Mat class_scores(1, num_classes, CV_32F, const_cast<float*>(d + 5));
        cv::Point max_loc;
        double max_val;
        cv::minMaxLoc(class_scores, nullptr, &max_val, nullptr, &max_loc);
        const float confidence = obj_conf * static_cast<float>(max_val);
        if (confidence < confidence_threshold_) continue;

        const float cx = d[0] * x_scale;
        const float cy = d[1] * y_scale;
        const float w  = d[2] * x_scale;
        const float h  = d[3] * y_scale;

        boxes.push_back({
            static_cast<int>(cx - w / 2),
            static_cast<int>(cy - h / 2),
            static_cast<int>(w),
            static_cast<int>(h)
        });
        confs.push_back(confidence);
        class_ids.push_back(max_loc.x);
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
        det.label      = (det.class_id < static_cast<int>(labels_.size()))
                             ? labels_[det.class_id] : "";
        results.push_back(std::move(det));
    }
    return results;
}

std::vector<Detection> DnnDetector::parse_ssd(cv::Mat& boxes_blob, cv::Mat& scores_blob,
                                               int orig_w, int orig_h) const {
    // boxes_blob:  [1, N, 4] — [y1, x1, y2, x2] normalized
    // scores_blob: [1, N, C] — per-class scores (index 0 = background)
    cv::Mat boxes_mat  = boxes_blob.reshape(1,  boxes_blob.size[1]);   // [N, 4]
    cv::Mat scores_mat = scores_blob.reshape(1, scores_blob.size[1]);  // [N, C]

    std::vector<cv::Rect> boxes;
    std::vector<float>    confs;
    std::vector<int>      class_ids;

    for (int i = 0; i < boxes_mat.rows; ++i) {
        const float* s = scores_mat.ptr<float>(i);
        const int num_classes = scores_mat.cols;

        // Skip background class (index 0)
        int   best_class = 1;
        float best_score = (num_classes > 1) ? s[1] : 0.0f;
        for (int c = 2; c < num_classes; ++c) {
            if (s[c] > best_score) { best_score = s[c]; best_class = c; }
        }
        if (best_score < confidence_threshold_) continue;

        const float* b = boxes_mat.ptr<float>(i);
        const float y1 = b[0] * orig_h;
        const float x1 = b[1] * orig_w;
        const float y2 = b[2] * orig_h;
        const float x2 = b[3] * orig_w;

        boxes.push_back({
            static_cast<int>(x1), static_cast<int>(y1),
            static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)
        });
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
        det.label      = (det.class_id < static_cast<int>(labels_.size()))
                             ? labels_[det.class_id] : "";
        results.push_back(std::move(det));
    }
    return results;
}

} // namespace improc::ml
