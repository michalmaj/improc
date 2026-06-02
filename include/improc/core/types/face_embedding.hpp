#pragma once
#include <opencv2/core.hpp>

namespace improc::core {

/**
 * @brief Typed wrapper for a face recognition embedding vector.
 *
 * Produced by RecognizeFace::embed(). The descriptor is typically
 * a (1 × 128) CV_32F row vector from SFace.
 *
 * Use cosine_similarity() to compare two embeddings.
 * Values range from -1 (opposite) to 1 (identical).
 */
struct FaceEmbedding {
    cv::Mat descriptor;

    bool empty() const { return descriptor.empty(); }

    float cosine_similarity(const FaceEmbedding& other) const {
        if (descriptor.empty() || other.descriptor.empty()) return 0.0f;
        cv::Mat n1, n2;
        cv::normalize(descriptor, n1);
        cv::normalize(other.descriptor, n2);
        return static_cast<float>(n1.dot(n2));
    }
};

} // namespace improc::core
