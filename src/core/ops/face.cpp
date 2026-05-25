// src/core/ops/face.cpp
#include "improc/core/ops/face.hpp"
#include <opencv2/core.hpp>
#include <filesystem>

namespace improc::core {

// ── DetectFaceYN ─────────────────────────────────────────────────────────────

void DetectFaceYN::ensure_initialized(cv::Size frame_size) {
    if (model_path_.empty())
        throw std::invalid_argument("DetectFaceYN: model path not set — call .model(path) first");
    if (!detector_ || frame_size != last_size_) {
        if (!std::filesystem::exists(model_path_))
            throw improc::FileNotFoundError{model_path_};
        try {
            detector_ = cv::FaceDetectorYN::create(
                model_path_, "", frame_size,
                score_threshold_, nms_threshold_, top_k_);
        } catch (const cv::Exception& e) {
            throw improc::ModelError{model_path_, e.what()};
        }
        last_size_ = frame_size;
    }
}

std::vector<FaceDetection> DetectFaceYN::operator()(Image<BGR> img) {
    ensure_initialized(img.mat().size());
    cv::Mat faces_mat;
    detector_->detect(img.mat(), faces_mat);
    std::vector<FaceDetection> out;
    if (faces_mat.empty() || faces_mat.rows == 0)
        return out;
    out.reserve(faces_mat.rows);
    for (int i = 0; i < faces_mat.rows; ++i) {
        const float* d = faces_mat.ptr<float>(i);
        FaceDetection fd;
        fd.bbox        = cv::Rect2f(d[0], d[1], d[2], d[3]);
        fd.confidence  = d[14];
        fd.landmarks[0] = {d[4],  d[5]};   // right eye
        fd.landmarks[1] = {d[6],  d[7]};   // left eye
        fd.landmarks[2] = {d[8],  d[9]};   // nose tip
        fd.landmarks[3] = {d[10], d[11]};  // right mouth corner
        fd.landmarks[4] = {d[12], d[13]};  // left mouth corner
        out.push_back(std::move(fd));
    }
    return out;
}

// ── RecognizeFace ─────────────────────────────────────────────────────────────

void RecognizeFace::ensure_initialized() {
    if (model_path_.empty())
        throw std::invalid_argument("RecognizeFace: model path not set — call .model(path) first");
    if (!recognizer_) {
        if (!std::filesystem::exists(model_path_))
            throw improc::FileNotFoundError{model_path_};
        try {
            recognizer_ = cv::FaceRecognizerSF::create(model_path_, "");
        } catch (const cv::Exception& e) {
            throw improc::ModelError{model_path_, e.what()};
        }
    }
}

cv::Mat RecognizeFace::embed(Image<BGR> face_chip) {
    ensure_initialized();
    cv::Mat emb;
    recognizer_->feature(face_chip.mat(), emb);
    return emb;
}

float RecognizeFace::match(const cv::Mat& emb_a, const cv::Mat& emb_b) {
    cv::Mat a_norm, b_norm;
    cv::normalize(emb_a, a_norm, 1.0, 0.0, cv::NORM_L2);
    cv::normalize(emb_b, b_norm, 1.0, 0.0, cv::NORM_L2);
    return static_cast<float>(a_norm.dot(b_norm));
}

} // namespace improc::core
