// include/improc/core/ops/face.hpp
#pragma once
#include <string>
#include <vector>
#include <opencv2/objdetect.hpp>
#include "improc/core/image.hpp"
#include "improc/core/ops/detector_types.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/// @brief Stateful YuNet face detector. Requires `.model(path)` before first call.
/// Hold as a named lvalue — operator() is non-const (lazy model init on first call).
/// Throws std::invalid_argument if model path not set; FileNotFoundError if absent; ModelError on load failure.
/// Setters `score_threshold`, `nms_threshold`, `top_k` are only applied at model creation;
/// calling them after the first `operator()` has no effect.
/// @code
/// DetectFaceYN op;
/// op.model("face_detection_yunet.onnx").score_threshold(0.9f);
/// auto faces = op(bgr_img);  // std::vector<FaceDetection>
/// @endcode
/// Fluent setters: `model()`, `score_threshold()`, `nms_threshold()`, `top_k()`.
class DetectFaceYN {
public:
    DetectFaceYN& model(std::string path)  { model_path_       = std::move(path); return *this; }
    DetectFaceYN& score_threshold(float t) { score_threshold_  = t;               return *this; }
    DetectFaceYN& nms_threshold(float t)   { nms_threshold_    = t;               return *this; }
    DetectFaceYN& top_k(int k)             { top_k_            = k;               return *this; }

    [[nodiscard]] std::vector<FaceDetection> operator()(Image<BGR> img);  // non-const: lazy init

private:
    void ensure_initialized(cv::Size frame_size);

    std::string model_path_;
    float       score_threshold_ = 0.9f;
    float       nms_threshold_   = 0.3f;
    int         top_k_           = 5000;

    cv::Ptr<cv::FaceDetectorYN> detector_;
    cv::Size                    last_size_ = {0, 0};
};

/// @brief Stateful SFace face recogniser. Requires `.model(path)` before calling embed().
/// embed() returns (1, 128) CV_32F embedding; non-const (lazy init).
/// match() is static — cosine similarity [-1, 1]; no model needed.
/// @code
/// RecognizeFace op;
/// op.model("face_recognition_sface.onnx");
/// auto emb = op.embed(face_chip);
/// float sim = RecognizeFace::match(emb_a, emb_b);
/// @endcode
class RecognizeFace {
public:
    RecognizeFace& model(std::string path) { model_path_ = std::move(path); return *this; }

    [[nodiscard]] cv::Mat      embed(Image<BGR> face_chip);           // non-const; returns (1, 128) CV_32F
    static float match(const cv::Mat& emb_a,
                       const cv::Mat& emb_b);           // static: cosine similarity, no model needed

private:
    void ensure_initialized();

    std::string                   model_path_;
    cv::Ptr<cv::FaceRecognizerSF> recognizer_;
};

} // namespace improc::core
